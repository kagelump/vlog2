"""TVAS Status GUI

A non-interactive GUI for monitoring the Travel Vlog Automation System progress.
"""

import asyncio
import logging
import threading
import functools
import tempfile
import io
import gc
import time
import json
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from shared import DEFAULT_VLM_MODEL, load_prompt, set_prompt_override, get_openrouter_api_key
from tvas.analysis import ClipAnalysis, analyze_clips_batch
from tvas.trim import detect_trims_batch
from shared.proxy import generate_proxies_batch

# Configure logging to capture everything
logger = logging.getLogger(__name__)


class GuiLogHandler(logging.Handler):
    """Custom logging handler that writes to a Toga Label."""
    
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    def emit(self, record):
        msg = self.format(record)
        self.app.loop.call_soon_threadsafe(self.update_log, msg)

    def update_log(self, msg):
        if hasattr(self.app, "log_label"):
            max_chars = 120
            
            if len(msg) > max_chars:
                msg = msg[:max_chars-3] + "..."
            self.app.log_label.text = msg


class SettingsWindow(toga.Window):
    def __init__(self, app_instance):
        super().__init__(title="Settings", size=(800, 600))
        self.app_instance = app_instance
        self.init_ui()

    def init_ui(self):
        # General Settings
        self.model_input = toga.TextInput(value=self.app_instance.model, style=Pack(flex=1))
        self.api_base_input = toga.TextInput(value=self.app_instance.api_base or "", style=Pack(flex=1))
        self.api_key_input = toga.TextInput(value=self.app_instance.api_key or "", style=Pack(flex=1))
        self.workers_input = toga.NumberInput(value=self.app_instance.max_workers, min=1, max=8, step=1, style=Pack(width=80))

        general_box = toga.Box(
            children=[
                toga.Box(children=[toga.Label("Model:", style=Pack(width=100)), self.model_input], style=Pack(direction=ROW, margin=5)),
                toga.Box(children=[toga.Label("API Base:", style=Pack(width=100)), self.api_base_input], style=Pack(direction=ROW, margin=5)),
                toga.Box(children=[toga.Label("API Key:", style=Pack(width=100)), self.api_key_input], style=Pack(direction=ROW, margin=5)),
                toga.Box(children=[toga.Label("Workers:", style=Pack(width=100)), self.workers_input], style=Pack(direction=ROW, margin=5)),
            ],
            style=Pack(direction=COLUMN, margin=10)
        )

        # Prompts
        self.prompt_inputs = {}
        prompt_files = [
            "video_describe.txt",
            "video_trim.txt",
        ]
        
        prompt_container = toga.OptionContainer(style=Pack(flex=1))
        
        for pf in prompt_files:
            try:
                content = load_prompt(pf)
            except:
                content = ""
            text_input = toga.MultilineTextInput(value=content, style=Pack(flex=1, font_family="monospace"))
            self.prompt_inputs[pf] = text_input
            tab_content = toga.Box(children=[text_input], style=Pack(flex=1, margin=5))
            prompt_container.content.append(toga.OptionItem(pf.replace(".txt", ""), tab_content))

        # Buttons
        save_btn = toga.Button("Apply", on_press=self.save_settings, style=Pack(margin=5))
        close_btn = toga.Button("Close", on_press=self.close_window, style=Pack(margin=5))
        
        btn_box = toga.Box(
            children=[
                toga.Box(style=Pack(flex=1)),
                save_btn, 
                close_btn
            ], 
            style=Pack(direction=ROW, margin=10)
        )

        self.content = toga.Box(
            children=[
                toga.Label("General Settings", style=Pack(font_weight='bold', margin=10)),
                general_box,
                toga.Label("Prompt Overrides (Session Only)", style=Pack(font_weight='bold', margin=10)),
                prompt_container,
                btn_box
            ],
            style=Pack(direction=COLUMN)
        )

    def save_settings(self, widget):
        self.app_instance.model = self.model_input.value
        self.app_instance.api_base = self.api_base_input.value if self.api_base_input.value.strip() else None
        self.app_instance.api_key = self.api_key_input.value
        self.app_instance.max_workers = int(self.workers_input.value)
        
        for pf, input_widget in self.prompt_inputs.items():
            set_prompt_override(pf, input_widget.value)
            
        self.app_instance.update_mode_label()
        self.app_instance.main_window.info_dialog("Settings", "Settings applied for this session.")
        self.close()

    def close_window(self, widget):
        self.close()


class ClipPreviewWindow(toga.Window):
    """Window showing video frame preview for a clip."""
    
    def __init__(self, app_instance, analysis: ClipAnalysis):
        super().__init__(title=f"Clip Preview: {analysis.source_path.name}", size=(800, 600))
        self.app_instance = app_instance
        self.analysis = analysis
        self.init_ui()

    def init_ui(self):
        """Initialize the clip preview UI."""
        image_view = None
        
        try:
            # Try to extract a frame from the video
            video_path = self.analysis.proxy_path or self.analysis.source_path
            if video_path.exists():
                # Extract a frame at the thumbnail timestamp or 1 second in
                timestamp = self.analysis.thumbnail_timestamp_sec or 1.0
                
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    temp_path = tf.name
                
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(timestamp),
                    "-i", str(video_path),
                    "-vframes", "1", "-q:v", "2",
                    temp_path
                ], capture_output=True, timeout=30)
                
                if Path(temp_path).exists():
                    with Image.open(temp_path) as img:
                        img = ImageOps.exif_transpose(img)
                        img_w, img_h = img.size
                        # Scale to fit window
                        scale = min(780 / img_w, 580 / img_h)
                        display_w = int(img_w * scale)
                        display_h = int(img_h * scale)
                        
                        image_view = toga.ImageView(
                            image=toga.Image(temp_path),
                            style=Pack(width=display_w, height=display_h)
                        )
        except Exception as e:
            logger.error(f"Failed to load clip preview: {e}")
            
        if image_view is None:
            image_view = toga.Label("Could not load preview", style=Pack(flex=1))

        scroll_container = toga.ScrollContainer(horizontal=True, vertical=True, style=Pack(flex=1))
        scroll_container.content = image_view
        
        self.content = scroll_container


class TvasStatusApp(toga.App):
    def __init__(
        self,
        directory: Optional[Path] = None,
        model: str = DEFAULT_VLM_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 1,
    ):
        super().__init__("TVAS Status", "com.tvas.tvas_status")
        self.directory = directory
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_workers = max_workers
        self.processed_count = 0
        self.total_count = 0
        self.recent_clips: list[ClipAnalysis] = []
        self.is_running = False
        self.is_review_mode = False
        self.stop_event = threading.Event()
        self.on_exit = self.exit_handler
        self.analysis_start_time = None
        self.initial_processed_count = None

    def exit_handler(self, app):
        """Handle app exit."""
        if self.is_running:
            logger.info("Stopping analysis...")
            self.stop_event.set()
        return True

    def update_mode_label(self):
        """Update the mode label based on current settings."""
        if hasattr(self, 'mode_label'):
            if self.api_base:
                self.mode_label.text = "[API MODE]"
                self.mode_label.style.color = "green"
            else:
                self.mode_label.text = "[MLX-VLM]"
                self.mode_label.style.color = "#D4AF37"  # Gold

    def startup(self):
        """Construct and show the Toga application."""
        
        # --- Control Panel: Folder Selection and Start Button ---
        self.folder_input = toga.TextInput(
            readonly=True,
            placeholder="Select a folder to scan...",
            style=Pack(flex=1, margin=(0, 5))
        )
        if self.directory:
            self.folder_input.value = str(self.directory)
        
        self.folder_button = toga.Button(
            "Browse...",
            on_press=self.select_folder,
            style=Pack(margin=(0, 5))
        )
        
        self.start_button = toga.Button(
            "Start Analysis",
            on_press=self.start_analysis,
            enabled=self.directory is not None,
            style=Pack(margin=(0, 5), color='blue')
        )
        
        self.settings_button = toga.Button(
            "Settings",
            on_press=self.open_settings,
            style=Pack(margin=(0, 5))
        )

        folder_row = toga.Box(
            children=[
                toga.Label("Folder:", style=Pack(margin=(5, 5), width=60)),
                self.folder_input,
                self.folder_button,
                self.settings_button,
                self.start_button
            ],
            style=Pack(direction=ROW, margin=5)
        )
        
        # --- Header: Progress & Logs ---
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(margin=(0, 10), flex=1))
        
        self.mode_label = toga.Label("", style=Pack(margin=(5, 5), font_weight='bold'))
        self.update_mode_label()
        
        self.status_label = toga.Label("Ready to start", style=Pack(margin=(5, 5), flex=1))
        
        status_row = toga.Box(
            children=[self.mode_label, self.status_label],
            style=Pack(direction=ROW)
        )

        self.log_label = toga.Label(
            "Select a folder and click Start Analysis", 
            style=Pack(margin=(0, 10), 
                       font_family="monospace", 
                       font_size=10, 
                       flex=1))
        
        self.resume_button = toga.Button(
            "Resume Live View",
            on_press=self.resume_live_view,
            enabled=False,
            style=Pack(margin=(0, 5))
        )

        log_row = toga.Box(
            children=[self.log_label, self.resume_button],
            style=Pack(direction=ROW, align_items=CENTER)
        )
        
        header_box = toga.Box(
            children=[status_row, self.progress_bar, log_row],
            style=Pack(direction=COLUMN, margin=10)
        )

        # --- Main: Current Clip & Details ---
        self.image_view = toga.ImageView(style=Pack(flex=1))
        
        self.clip_label = toga.Label("No clip loaded", style=Pack(margin=5, text_align=CENTER))
        
        self.image_area = toga.Box(
            children=[self.image_view, self.clip_label],
            style=Pack(direction=COLUMN, flex=1)
        )

        # Details Panel
        self.details_label = toga.Label("Details", style=Pack(font_weight='bold', margin_bottom=5))
        self.details_content = toga.MultilineTextInput(readonly=True, style=Pack(flex=1))
        self.preview_btn = toga.Button("Preview Frame", on_press=self.open_preview, enabled=False, style=Pack(margin_top=5))
        
        self.details_panel = toga.Box(
            children=[self.details_label, self.details_content, self.preview_btn],
            style=Pack(direction=COLUMN, width=350, margin=10)
        )
        
        main_box = toga.Box(
            children=[self.image_area],
            style=Pack(direction=ROW, flex=1, margin=10)
        )
        self.main_box = main_box

        # --- Footer: Recent Clips ---
        self.recent_box = toga.Box(style=Pack(direction=ROW, margin=10))
        
        self.recent_scroll = toga.ScrollContainer(
            horizontal=True,
            vertical=False,
            style=Pack(height=180, flex=1)
        )
        self.recent_scroll.content = self.recent_box
        
        footer_container = toga.Box(
            children=[toga.Label("Recent Processed", style=Pack(margin=5)), self.recent_scroll],
            style=Pack(direction=COLUMN)
        )

        # --- Main Layout ---
        self.main_window = toga.MainWindow(title=self.formal_name, size=(1100, 800))
        self.main_window.content = toga.Box(
            children=[folder_row, header_box, main_box, footer_container],
            style=Pack(direction=COLUMN)
        )
        
        # Setup Logging
        handler = GuiLogHandler(self)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.main_window.show()

        # Attempt to maximize window
        try:
            self.main_window.state = toga.WindowState.MAXIMIZED
        except AttributeError:
            try:
                if hasattr(self, 'screens') and self.screens:
                    screen = self.screens[0]
                    self.main_window.size = (screen.size.width, screen.size.height)
                    self.main_window.position = (0, 0)
            except Exception as e:
                logger.warning(f"Failed to maximize window: {e}")

        if self.directory:
            self.on_running = self.auto_start_analysis

    async def auto_start_analysis(self, app):
        """Automatically start analysis if directory is provided."""
        await asyncio.sleep(0.5)
        await self.load_existing_analyses()
        await self.start_analysis(self.start_button)
    
    def open_settings(self, widget):
        """Open the settings window."""
        settings_window = SettingsWindow(self)
        settings_window.show()

    async def load_existing_analyses(self):
        """Load existing JSON analysis files from the directory."""
        if not self.directory:
            return

        self.status_label.text = "Checking for existing analysis files..."
        
        loop = asyncio.get_running_loop()
        
        def _load():
            analyses = []
            # Look for individual JSON files (not analysis.json)
            for json_file in self.directory.glob("*.json"):
                if json_file.name == "analysis.json":
                    continue
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if 'metadata' in data:
                        meta = data['metadata']
                        analysis = ClipAnalysis(
                            source_path=Path(meta.get('source_path', str(json_file))),
                            proxy_path=Path(meta.get('proxy_path', '')) if meta.get('proxy_path') else None,
                            duration_seconds=meta.get('duration_seconds', 0),
                            clip_name=data.get('clip_name'),
                            clip_description=data.get('clip_description'),
                            audio_description=data.get('audio_description'),
                            subject_keywords=data.get('subject_keywords', []),
                            action_keywords=data.get('action_keywords', []),
                            time_of_day=data.get('time_of_day'),
                            environment=data.get('environment'),
                            mood=data.get('mood'),
                            people_presence=data.get('people_presence'),
                            thumbnail_timestamp_sec=data.get('thumbnail_timestamp_sec'),
                            beat_id=data.get('beat_id'),
                            beat_title=data.get('beat_title'),
                            needs_trim=data.get('needs_trim', False),
                            suggested_in_point=data.get('suggested_in_point'),
                            suggested_out_point=data.get('suggested_out_point'),
                        )
                        analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to load analysis from {json_file}: {e}")
            return analyses

        existing_analyses = await loop.run_in_executor(None, _load)
        
        if existing_analyses:
            logger.info(f"Loaded {len(existing_analyses)} existing analysis files.")
            self.status_label.text = f"Loaded {len(existing_analyses)} existing analyses. Press 'Start Analysis' to continue."
            
            for analysis in existing_analyses:
                self.add_recent_clip(analysis)
                await asyncio.sleep(0.01)
        else:
            self.status_label.text = "No existing analysis files found."

    async def select_folder(self, widget):
        """Handle folder selection."""
        try:
            folder = await self.main_window.select_folder_dialog(
                title="Select folder to scan for videos"
            )
            
            if folder:
                self.directory = Path(folder)
                self.folder_input.value = str(self.directory)
                self.start_button.enabled = True
                self.status_label.text = "Folder selected. Click Start Analysis to begin."
                logger.info(f"Selected folder: {self.directory}")
                
                await self.load_existing_analyses()
        except Exception as e:
            logger.error(f"Error selecting folder: {e}")
            self.status_label.text = f"Error selecting folder. Please try again."
    
    async def start_analysis(self, widget):
        """Start the analysis when user clicks the button."""
        if not self.directory:
            self.status_label.text = "Please select a folder first."
            return
        
        if self.is_running:
            self.status_label.text = "Analysis is already running."
            return
        
        self.start_button.enabled = False
        self.folder_button.enabled = False
        self.stop_event.clear()
        
        gc.collect()
        
        try:
            await self.run_analysis(widget)
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            self.status_label.text = f"Analysis failed: {e}"
            self.start_button.enabled = True
            self.folder_button.enabled = True
            self.is_running = False

    async def run_analysis(self, widget):
        """Run the analysis in a background thread."""
        self.is_running = True
        self.status_label.text = f"Scanning {self.directory}..."
        self.analysis_start_time = time.time()
        self.initial_processed_count = None
        
        loop = asyncio.get_running_loop()
        
        try:
            # Find video files
            video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mxf', '.MXF', '.mts', '.MTS', '.insv', '.INSV', '.insp', '.INSP'}
            
            def find_videos():
                videos = []
                for video_file in self.directory.iterdir():
                    if video_file.is_file() and video_file.suffix in video_extensions:
                        videos.append(video_file)
                return sorted(videos)
            
            videos = await loop.run_in_executor(None, find_videos)
            
            if not videos:
                self.status_label.text = "No video files found."
                self.is_running = False
                self.start_button.enabled = True
                self.folder_button.enabled = True
                return

            self.total_count = len(videos)
            self.progress_bar.max = self.total_count
            self.status_label.text = f"Found {self.total_count} videos. Starting analysis..."

            # Process videos
            await loop.run_in_executor(
                None,
                self._process_videos,
                videos,
            )
            
            if self.stop_event.is_set():
                self.status_label.text = "Processing cancelled."
            else:
                self.status_label.text = "Processing Complete!"
                self.progress_bar.value = self.total_count
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            self.status_label.text = f"Error: {e}"
        finally:
            self.is_running = False
            self.start_button.enabled = True
            self.folder_button.enabled = True

    def _process_videos(self, videos: list[Path]):
        """Process video files - runs in executor."""
        clips_to_analyze = [(v, None) for v in videos]
        
        try:
            # Run analysis
            analyses = analyze_clips_batch(
                clips_to_analyze,
                use_vlm=True,
                model_name=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                provider_preferences=None,
                max_workers=self.max_workers,
            )
            
            # Update UI with results after completion
            for i, analysis in enumerate(analyses):
                if self.stop_event.is_set():
                    break
                self.loop.call_soon_threadsafe(
                    self.update_ui, i + 1, len(analyses), analysis
                )
            
            # Run trim detection if we have a directory
            if self.directory and not self.stop_event.is_set():
                logger.info("Running trim detection...")
                detect_trims_batch(
                    project_dir=self.directory,
                    model_name=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    provider_preferences=None,
                    max_workers=self.max_workers,
                )
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _extract_thumbnail(self, video_path: Path, timestamp: float = 1.0) -> Optional[bytes]:
        """Extract a thumbnail from a video file."""
        import subprocess
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                temp_path = tf.name
            
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1", "-q:v", "2",
                "-vf", "scale=320:-1",
                temp_path
            ], capture_output=True, timeout=30)
            
            if Path(temp_path).exists():
                with open(temp_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to extract thumbnail: {e}")
        return None

    def load_preview_image(self, path: Path, timestamp: float = 1.0) -> Optional[toga.Image]:
        """Load a preview image from a video file."""
        try:
            thumb_bytes = self._extract_thumbnail(path, timestamp)
            if thumb_bytes:
                return toga.Image(src=thumb_bytes)
        except Exception as e:
            logger.warning(f"Failed to load preview {path}: {e}")
        return None

    def update_ui(self, processed: int, total: int, analysis: ClipAnalysis):
        """Update UI elements on the main thread."""
        self.processed_count = processed
        self.progress_bar.value = processed
        
        # Calculate stats
        if self.initial_processed_count is None:
            self.initial_processed_count = processed
            
        delta_processed = processed - self.initial_processed_count
        
        avg_speed = 30.0  # Default estimate
        if delta_processed > 0 and self.analysis_start_time:
            elapsed = time.time() - self.analysis_start_time
            avg_speed = elapsed / delta_processed
            
        remaining = total - processed
        eta_seconds = remaining * avg_speed
        
        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        else:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            
        status_text = f"Processing: {processed}/{total} | Avg: {avg_speed:.1f}s/clip | ETA: {eta_str}"
        self.status_label.text = status_text
        
        # Update main view if not in review mode
        if not self.is_review_mode and analysis:
            self.clip_label.text = analysis.source_path.name
            
            # Load preview thumbnail
            video_path = analysis.proxy_path or analysis.source_path
            timestamp = analysis.thumbnail_timestamp_sec or 1.0
            
            try:
                img = self.load_preview_image(video_path, timestamp)
                if img:
                    self.image_view.image = img
            except Exception as e:
                logger.warning(f"Failed to load preview: {e}")

        if analysis:
            self.add_recent_clip(analysis)

    def open_preview(self, widget):
        """Open preview window for current clip."""
        if hasattr(self, 'current_review_analysis') and self.current_review_analysis:
            window = ClipPreviewWindow(self, self.current_review_analysis)
            window.show()

    def show_details(self, analysis: ClipAnalysis):
        """Show details for a specific clip."""
        self.is_review_mode = True
        self.resume_button.enabled = True
        self.current_review_analysis = analysis
        self.preview_btn.enabled = True
        
        # Update main image
        try:
            video_path = analysis.proxy_path or analysis.source_path
            timestamp = analysis.thumbnail_timestamp_sec or 1.0
            
            img = self.load_preview_image(video_path, timestamp)
            if img:
                self.image_view.image = img
            
            self.clip_label.text = analysis.source_path.name
        except Exception as e:
            logger.error(f"Failed to load preview for details: {e}")

        # Update details panel
        details_text = f"Clip: {analysis.clip_name or analysis.source_path.name}\n\n"
        details_text += f"Duration: {analysis.duration_seconds:.1f}s\n\n"
        
        if analysis.clip_description:
            details_text += f"Description:\n{analysis.clip_description}\n\n"
        
        if analysis.audio_description:
            details_text += f"Audio:\n{analysis.audio_description}\n\n"
        
        if analysis.subject_keywords:
            details_text += f"Subjects: {', '.join(analysis.subject_keywords)}\n\n"
        
        if analysis.action_keywords:
            details_text += f"Actions: {', '.join(analysis.action_keywords)}\n\n"
        
        if analysis.time_of_day:
            details_text += f"Time of Day: {analysis.time_of_day}\n"
        
        if analysis.environment:
            details_text += f"Environment: {analysis.environment}\n"
        
        if analysis.mood:
            details_text += f"Mood: {analysis.mood}\n"
        
        if analysis.people_presence:
            details_text += f"People: {analysis.people_presence}\n"
        
        if analysis.needs_trim:
            details_text += f"\n--- Trim Suggestion ---\n"
            if analysis.suggested_in_point is not None:
                details_text += f"In: {analysis.suggested_in_point:.2f}s\n"
            if analysis.suggested_out_point is not None:
                details_text += f"Out: {analysis.suggested_out_point:.2f}s\n"
        
        if analysis.beat_title:
            details_text += f"\n--- Beat Assignment ---\n"
            details_text += f"Beat: {analysis.beat_title}\n"
            if analysis.beat_classification:
                details_text += f"Role: {analysis.beat_classification}\n"
            if analysis.beat_reasoning:
                details_text += f"Reason: {analysis.beat_reasoning}\n"

        self.details_content.value = details_text
        
        # Show details panel if not visible
        if self.details_panel not in self.main_box.children:
            self.main_box.add(self.details_panel)

    def resume_live_view(self, widget):
        """Resume live view updates."""
        self.is_review_mode = False
        self.resume_button.enabled = False
        self.current_review_analysis = None
        self.preview_btn.enabled = False
        
        # Hide details panel
        if self.details_panel in self.main_box.children:
            self.main_box.remove(self.details_panel)
            
        self.clip_label.text = "Resuming live view..."

    def add_recent_clip(self, analysis: ClipAnalysis):
        """Add a thumbnail to the recent clips strip."""
        thumb_box = toga.Box(style=Pack(direction=COLUMN, width=140, margin=5))
        
        try:
            # Get thumbnail
            video_path = analysis.proxy_path or analysis.source_path
            timestamp = analysis.thumbnail_timestamp_sec or 1.0
            thumb_bytes = self._extract_thumbnail(video_path, timestamp)
            
            if thumb_bytes:
                toga_img = toga.Image(src=thumb_bytes)
                image_view = toga.ImageView(image=toga_img, style=Pack(height=80, width=120))
                thumb_box.add(image_view)
            
            # Add View button
            view_btn = toga.Button(
                "View", 
                on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
                style=Pack(width=120)
            )
            thumb_box.add(view_btn)
            
        except Exception as e:
            logger.warning(f"Failed to create thumbnail view: {e}")
            view_widget = toga.Button(
                "View", 
                on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
                style=Pack(height=80, width=120)
            )
            thumb_box.add(view_widget)
        
        # Clip name label
        clip_name = analysis.clip_name or analysis.source_path.stem
        if len(clip_name) > 18:
            clip_name = clip_name[:16] + ".."
        name_label = toga.Label(clip_name, style=Pack(text_align=CENTER, font_size=10))
        thumb_box.add(name_label)
        
        # Duration label
        duration_label = toga.Label(f"{analysis.duration_seconds:.1f}s", style=Pack(text_align=CENTER, font_size=9))
        thumb_box.add(duration_label)
        
        # Add to start of list
        self.recent_box.insert(0, thumb_box)


def main(
    directory: Optional[Path] = None,
    model: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_workers: int = 1,
):
    """Create and return the TVAS Status App."""
    return TvasStatusApp(directory, model, api_base, api_key, max_workers)
