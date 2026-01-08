import json
import logging
import base64
import urllib.request
import urllib.error
import subprocess
import time
import tempfile
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Any, Tuple, Sequence, cast
from threading import Lock

from shared.ffmpeg_utils import detect_best_video_codec

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks API costs across the application lifecycle."""
    _total_cost: float = 0.0
    _lock: Lock = Lock()

    @classmethod
    def add(cls, amount: float):
        if amount is None:
            return
        with cls._lock:
            cls._total_cost += float(amount)

    @classmethod
    def get_total(cls) -> float:
        with cls._lock:
            return cls._total_cost

    @classmethod
    def report(cls):
        total = cls.get_total()
        if total > 0:
            # Using 6 decimal places as costs can be very small
            print(f"\n💰 Total API Cost: ${total:.6f}")

# Register report on exit
atexit.register(CostTracker.report)

@dataclass
class VLMResponse:
    """Response from VLM API."""
    text: str
    provider: Optional[str] = None
    cost: float = 0.0

def check_lmstudio_running():
    """Check if LM Studio is running locally, or try to start it."""
    # Check if already running
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=0.2) as response:
            if response.status == 200:
                return True
    except Exception:
        pass

    # Try to start via lms CLI
    try:
        logger.info("LM Studio server not detected. Attempting to start via 'lms server start'...")
        # Start in background
        subprocess.Popen(["lms", "server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait up to 5 seconds for it to become available
        for _ in range(10):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=0.2) as response:
                    if response.status == 200:
                        logger.info("LM Studio server started successfully.")
                        return True
            except Exception:
                pass
    except FileNotFoundError:
        logger.debug("'lms' command not found.")
    except Exception as e:
        logger.debug(f"Failed to start LM Studio: {e}")

    return False

class VLMClient:
    """Client for interacting with Vision Language Models (local or API)."""

    def __init__(
        self,
        model_name: str = "local-model",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        provider_preferences: Optional[str] = None,
        app_name: str = "tvas"
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.provider_preferences = provider_preferences
        self.app_name = app_name
        self.model = None
        self.processor = None
        self.config = None

        if not self.api_base:
            self._load_local_model()

    def _load_local_model(self):
        """Load the local mlx-vlm model."""
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            
            logger.info(f"Loading mlx-vlm model: {self.model_name}")
            self.model, self.processor = load(self.model_name, trust_remote_code=True)
            self.config = load_config(self.model_name, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Optional[VLMResponse]:
        """Generate response from VLM using images."""
        if self.api_base:
            return self._call_api(prompt, image_paths, max_tokens, temperature)
        else:
            return self._generate_local(prompt, image_paths, max_tokens, temperature)

    def generate_from_video(
        self,
        prompt: str,
        video_path: Path,
        fps: float = 1.0,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_pixels: int = 224 * 224,
        transcription: Optional[str] = None
    ) -> Optional[VLMResponse]:
        """Generate response from VLM using a video file."""
        if self.api_base:
            # Use ffmpeg to process video for API
            base64_video = self._process_video_ffmpeg(video_path, fps)
            if not base64_video:
                return None
            
            return self._call_api(
                prompt, 
                [base64_video], 
                max_tokens, 
                temperature, 
                is_base64=True,
                media_type="video/mp4",
                transcription=transcription
            )
        else:
            return self._generate_local_video(prompt, video_path, fps, max_tokens, temperature, max_pixels, transcription)

    def _process_video_ffmpeg(self, video_path: Path, fps: float) -> Optional[str]:
        """Process video using ffmpeg and return base64 string."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = Path(temp_file.name)
            
        try:
            codec_flags = detect_best_video_codec()
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps={fps},scale='min(768,iw)':-2"
            ] + codec_flags + [
                "-an",
                str(temp_output)
            ]
            
            logger.info(f"Running ffmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None
                
            # Read and encode
            with open(temp_output, "rb") as f:
                video_data = f.read()
                return base64.b64encode(video_data).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to process video with ffmpeg: {e}")
            return None
        finally:
            if temp_output.exists():
                try:
                    temp_output.unlink()
                except:
                    pass



    def _call_api(
        self,
        prompt: str,
        media_items: Union[list[Path], list[str]],
        max_tokens: int,
        temperature: float,
        is_base64: bool = False,
        media_type: str = "image/jpeg",
        transcription: Optional[str] = None
    ) -> Optional[VLMResponse]:
        """Call a VLM API (OpenAI compatible)."""
        try:
            messages_content: List[dict[str, Any]] = [{"type": "text", "text": prompt}]
            
            if transcription:
                messages_content.append({"type": "text", "text": f"Transcription:\n{transcription}"})
            
            for item in media_items:
                if is_base64:
                    base64_data = item
                else:
                    with open(item, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode('utf-8')
                
                if media_type.startswith("video/"):
                    messages_content.append({
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{media_type};base64,{base64_data}"
                        }
                    })
                else:
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_data}"
                        }
                    })

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/kagelump/vlog2", # Identify the app source
                "X-Title": self.app_name.upper() # Show app name (TVAS/TPRS) in rankings
            }
            
            payload = {
                "model": self.model_name, 
                "messages": [
                    {
                        "role": "user",
                        "content": messages_content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if self.provider_preferences:
                # Handle comma-separated list
                providers = [p.strip() for p in self.provider_preferences.split(",") if p.strip()]
                if providers:
                    payload["provider"] = {"order": providers}
            
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                logging.debug(f"VLM Request: API response: {response_data}")
                if 'error' in response_data:
                    logger.error(f"API error: {response_data['error']}")
                    return None
                text = response_data['choices'][0]['message']['content']
                provider = response_data.get('provider')
                
                # Extract cost from usage if available (OpenRouter format)
                cost = response_data.get("usage", {}).get("cost", 0.0)
                CostTracker.add(cost)
                
                return VLMResponse(text=text, provider=provider, cost=cost)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"API call failed with HTTP {e.code}: {e.reason}")
            logger.error(f"Server response: {error_body}")
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def _generate_local(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int,
        temperature: float
    ) -> Optional[VLMResponse]:
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            if not self.model or not self.processor:
                logger.error("Model not loaded")
                return None

            image_paths_str = [str(p) for p in image_paths]
            formatted_prompt = apply_chat_template(
                self.processor, self.config, prompt, num_images=len(image_paths)
            )
            
            response = generate(
                self.model,
                self.processor,
                formatted_prompt,
                image_paths_str,
                verbose=False,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            if hasattr(response, "text"):
                text = response.text
            else:
                text = str(response)
            
            return VLMResponse(text=text, provider="mlx-vlm")
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return None

    def _generate_local_video(
        self,
        prompt: str,
        video_path: Path,
        fps: float,
        max_tokens: int,
        temperature: float,
        max_pixels: int,
        transcription: Optional[str] = None
    ) -> Optional[VLMResponse]:
        try:
            import mlx.core as mx
            from mlx_vlm import generate
            from mlx_vlm.video_generate import process_vision_info

            if not self.model or not self.processor:
                logger.error("Model not loaded")
                return None

            content = [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ]
            
            if transcription:
                content.append({"type": "text", "text": f"Transcription:\n{transcription}"})

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            _, video_inputs = cast(Tuple[Sequence[Any], Sequence[Any]], process_vision_info(messages))

            inputs = self.processor(
                text=[text],
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                video_metadata={'fps': fps, 'total_num_frames': video_inputs[0].shape[0]}
            )

            input_ids = mx.array(inputs["input_ids"])
            mask = mx.array(inputs["attention_mask"])
            video_grid_thw = mx.array(inputs["video_grid_thw"])

            extra = {"video_grid_thw": video_grid_thw}

            pixel_values = inputs.get(
                "pixel_values_videos", inputs.get("pixel_values", None)
            )
            
            if pixel_values is None:
                raise ValueError("Please provide a valid video input.")
            pixel_values = mx.array(pixel_values)

            response = generate(
                model=self.model,
                processor=self.processor,
                prompt=text,
                input_ids=input_ids,
                pixel_values=pixel_values,
                mask=mask,
                max_tokens=max_tokens,
                temp=temperature,
                **extra,
            )
            
            if hasattr(response, "text"):
                text = response.text
            else:
                text = str(response)

            return VLMResponse(text=text, provider="mlx-vlm")

        except Exception as e:
            logger.error(f"Local video generation failed: {e}")
            return None
