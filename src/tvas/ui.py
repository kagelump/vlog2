"""TVAS Status GUI

A GUI for monitoring and controlling the Travel Vlog Automation System.
Supports all phases: Ingestion, Proxy Generation, Analysis, Trim Detection, Beat Alignment.
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
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from PIL import Image, ImageOps
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from shared import DEFAULT_VLM_MODEL, load_prompt, set_prompt_override, get_openrouter_api_key
from tvas.analysis import ClipAnalysis, analyze_clips_batch
from tvas.trim import detect_trims_batch
from tvas.beats import align_beats
from tvas.ingestion import CameraType, detect_camera_type, ingest_volume, get_video_files
from tvas.watcher import find_camera_volumes, is_camera_volume
from shared.proxy import generate_proxies_batch

# Configure logging
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
            max_chars = 150
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
        prompt_files = ["video_describe.txt", "video_trim.txt"]
        
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
            children=[toga.Box(style=Pack(flex=1)), save_btn, close_btn], 
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
        self.app_instance.api_base = self.api_base_input.value.strip() or None
        self.app_instance.api_key = self.api_key_input.value
        self.app_instance.max_workers = int(self.workers_input.value or 1)
        
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
            video_path = self.analysis.proxy_path or self.analysis.source_path
            if video_path.exists():
                timestamp = self.analysis.thumbnail_timestamp_sec or 1.0
                
                import subprocess
                
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
        sd_card_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        proxy_path: Optional[Path] = None,
        model: str = DEFAULT_VLM_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 1,
    ):
        super().__init__("TVAS", "com.tvas.tvas_status")
        
        # Path settings
        self.sd_card_path = sd_card_path
        self.project_path = project_path
        self.proxy_path = proxy_path or Path.home() / "Movies" / "Vlog"
        
        # Model/API settings
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_workers = max_workers
        
        # State
        self.processed_count = 0
        self.total_count = 0
        self.recent_clips: list[ClipAnalysis] = []
        self.is_running = False
        self.is_review_mode = False
        self.stop_event = threading.Event()
        self.on_exit = self.exit_handler
        self.analysis_start_time = None
        self.project_name: Optional[str] = None
        self.outline_path: Optional[Path] = None

    def exit_handler(self, app):
        """Handle app exit."""
        if self.is_running:
            logger.info("Stopping processing...")
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
                self.mode_label.style.color = "#D4AF37"

    def _update_button_states(self):
        """Update button enabled states based on path availability."""
        has_sd = self.sd_card_path is not None and self.sd_card_path.exists()
        has_project = self.project_path is not None and self.project_path.exists()
        has_proxy = self.proxy_path is not None
        
        # Determine proxy directory
        proxy_dir = None
        if has_proxy and self.project_name:
            proxy_dir = self.proxy_path / self.project_name / "proxy"
        elif has_project:
            proxy_dir = self.proxy_path / self.project_path.name / "proxy"
        
        has_proxies = proxy_dir and proxy_dir.exists() and any(proxy_dir.glob("*.mp4"))
        has_analyses = proxy_dir and proxy_dir.exists() and any(proxy_dir.glob("*.json"))
        has_outline = self.outline_path is not None and self.outline_path.exists()
        
        running = self.is_running
        
        # Stage 1: Copy from SD card - needs SD card and project path
        self.copy_btn.enabled = has_sd and has_project and not running
        
        # Stage 2: Generate Proxies - needs project folder with video files
        self.proxy_btn.enabled = has_project and not running
        
        # Stage 3: Analysis - needs proxy folder with videos or project folder
        self.analyze_btn.enabled = (has_proxies or has_project) and not running
        
        # Stage 3.5: Trim Detection - needs analysis JSON files
        self.trim_btn.enabled = has_analyses and not running
        
        # Stage 4: Beat Alignment - needs analyses and outline
        self.beats_btn.enabled = has_analyses and has_outline and not running
        
        # Run All - needs at least project folder
        self.run_all_btn.enabled = (has_sd or has_project) and not running

    def startup(self):
        """Construct and show the Toga application."""
        
        # === PATH SELECTION SECTION ===
        path_section_label = toga.Label("Paths", style=Pack(font_weight='bold', margin=(10, 10, 5, 10)))
        
        # SD Card Volume
        self.sd_input = toga.TextInput(
            readonly=True,
            placeholder="Auto-detect or select SD card...",
            style=Pack(flex=1, margin=(0, 5))
        )
        self.sd_browse_btn = toga.Button("Browse...", on_press=self.select_sd_card, style=Pack(margin=(0, 5)))
        self.sd_detect_btn = toga.Button("Detect", on_press=self.detect_sd_card, style=Pack(margin=(0, 5)))
        
        sd_row = toga.Box(
            children=[
                toga.Label("SD Card:", style=Pack(margin=(5, 5), width=100)),
                self.sd_input,
                self.sd_browse_btn,
                self.sd_detect_btn,
            ],
            style=Pack(direction=ROW, margin=5)
        )
        
        # Project Folder (archival storage)
        self.project_input = toga.TextInput(
            readonly=True,
            placeholder="Project folder (e.g. /Volumes/Acasis/project_name)...",
            style=Pack(flex=1, margin=(0, 5))
        )
        self.project_browse_btn = toga.Button("Browse...", on_press=self.select_project_folder, style=Pack(margin=(0, 5)))
        self.project_detect_btn = toga.Button("Detect", on_press=self.detect_project_folder, style=Pack(margin=(0, 5)))
        
        project_row = toga.Box(
            children=[
                toga.Label("Project:", style=Pack(margin=(5, 5), width=100)),
                self.project_input,
                self.project_browse_btn,
                self.project_detect_btn,
            ],
            style=Pack(direction=ROW, margin=5)
        )
        
        # Proxy Folder
        self.proxy_input = toga.TextInput(
            readonly=True,
            placeholder="Proxy folder (default: ~/Movies/Vlog)...",
            style=Pack(flex=1, margin=(0, 5))
        )
        if self.proxy_path:
            self.proxy_input.value = str(self.proxy_path)
        self.proxy_browse_btn = toga.Button("Browse...", on_press=self.select_proxy_folder, style=Pack(margin=(0, 5)))
        
        proxy_row = toga.Box(
            children=[
                toga.Label("Proxy:", style=Pack(margin=(5, 5), width=100)),
                self.proxy_input,
                self.proxy_browse_btn,
            ],
            style=Pack(direction=ROW, margin=5)
        )
        
        # Outline file (for beat alignment)
        self.outline_input = toga.TextInput(
            readonly=True,
            placeholder="Optional: outline.md for beat alignment...",
            style=Pack(flex=1, margin=(0, 5))
        )
        self.outline_browse_btn = toga.Button("Browse...", on_press=self.select_outline_file, style=Pack(margin=(0, 5)))
        
        outline_row = toga.Box(
            children=[
                toga.Label("Outline:", style=Pack(margin=(5, 5), width=100)),
                self.outline_input,
                self.outline_browse_btn,
            ],
            style=Pack(direction=ROW, margin=5)
        )
        
        path_box = toga.Box(
            children=[path_section_label, sd_row, project_row, proxy_row, outline_row],
            style=Pack(direction=COLUMN, margin=5)
        )
        
        # === PHASE BUTTONS SECTION ===
        phase_section_label = toga.Label("Pipeline Phases", style=Pack(font_weight='bold', margin=(10, 10, 5, 10)))
        
        self.copy_btn = toga.Button(
            "1. Copy from SD",
            on_press=self.run_copy_phase,
            enabled=False,
            style=Pack(margin=5, flex=1)
        )
        self.proxy_btn = toga.Button(
            "2. Generate Proxies",
            on_press=self.run_proxy_phase,
            enabled=False,
            style=Pack(margin=5, flex=1)
        )
        self.analyze_btn = toga.Button(
            "3. AI Analysis",
            on_press=self.run_analysis_phase,
            enabled=False,
            style=Pack(margin=5, flex=1)
        )
        self.trim_btn = toga.Button(
            "3.5 Trim Detection",
            on_press=self.run_trim_phase,
            enabled=False,
            style=Pack(margin=5, flex=1)
        )
        self.beats_btn = toga.Button(
            "4. Beat Alignment",
            on_press=self.run_beats_phase,
            enabled=False,
            style=Pack(margin=5, flex=1)
        )
        
        phase_row1 = toga.Box(
            children=[self.copy_btn, self.proxy_btn, self.analyze_btn],
            style=Pack(direction=ROW, margin=5)
        )
        phase_row2 = toga.Box(
            children=[self.trim_btn, self.beats_btn],
            style=Pack(direction=ROW, margin=5)
        )
        
        # Run All and Settings
        self.run_all_btn = toga.Button(
            "▶ Run All Phases",
            on_press=self.run_all_phases,
            enabled=False,
            style=Pack(margin=5, flex=1, color='blue')
        )
        self.settings_btn = toga.Button(
            "Settings",
            on_press=self.open_settings,
            style=Pack(margin=5)
        )
        
        action_row = toga.Box(
            children=[self.run_all_btn, self.settings_btn],
            style=Pack(direction=ROW, margin=5)
        )
        
        phase_box = toga.Box(
            children=[phase_section_label, phase_row1, phase_row2, action_row],
            style=Pack(direction=COLUMN, margin=5)
        )
        
        # === PROGRESS SECTION ===
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(margin=(0, 10), flex=1))
        
        self.mode_label = toga.Label("", style=Pack(margin=(5, 5), font_weight='bold'))
        self.update_mode_label()
        
        self.status_label = toga.Label("Ready", style=Pack(margin=(5, 5), flex=1))
        
        status_row = toga.Box(
            children=[self.mode_label, self.status_label],
            style=Pack(direction=ROW)
        )

        self.log_label = toga.Label(
            "Configure paths and click a phase button to start", 
            style=Pack(margin=(0, 10), font_family="monospace", font_size=10, flex=1)
        )
        
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
        
        progress_box = toga.Box(
            children=[status_row, self.progress_bar, log_row],
            style=Pack(direction=COLUMN, margin=10)
        )

        # === MAIN CONTENT: Clip View & Details ===
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

        # === FOOTER: Recent Clips ===
        self.recent_box = toga.Box(style=Pack(direction=ROW, margin=10))
        
        self.recent_scroll = toga.ScrollContainer(
            horizontal=True,
            vertical=False,
            style=Pack(height=180, flex=1)
        )
        self.recent_scroll.content = self.recent_box
        
        footer_container = toga.Box(
            children=[toga.Label("Recent Clips", style=Pack(margin=5)), self.recent_scroll],
            style=Pack(direction=COLUMN)
        )

        # === MAIN LAYOUT ===
        self.main_window = toga.MainWindow(title=self.formal_name, size=(1200, 900))
        self.main_window.content = toga.Box(
            children=[path_box, phase_box, progress_box, main_box, footer_container],
            style=Pack(direction=COLUMN)
        )
        
        # Setup Logging
        handler = GuiLogHandler(self)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.main_window.show()

        # Maximize window
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

        # Auto-detect paths on startup
        self.on_running = self.auto_detect_paths

    async def auto_detect_paths(self, app):
        """Auto-detect paths on startup."""
        await asyncio.sleep(0.3)
        await self.detect_sd_card(None)
        await self.detect_project_folder(None)
        self._update_button_states()

    # === PATH SELECTION HANDLERS ===
    
    async def select_sd_card(self, widget):
        """Select SD card folder manually."""
        try:
            folder = await self.main_window.select_folder_dialog(title="Select SD card volume")
            if folder:
                self.sd_card_path = Path(folder)
                self.sd_input.value = str(self.sd_card_path)
                
                # Detect camera type
                if is_camera_volume(self.sd_card_path):
                    camera_type = detect_camera_type(self.sd_card_path)
                    logger.info(f"Selected SD card: {self.sd_card_path} ({camera_type.value})")
                else:
                    logger.warning(f"Selected folder doesn't appear to be a camera volume")
                
                self._update_button_states()
        except Exception as e:
            logger.error(f"Error selecting SD card: {e}")

    async def detect_sd_card(self, widget):
        """Auto-detect camera SD cards."""
        try:
            loop = asyncio.get_running_loop()
            camera_volumes = await loop.run_in_executor(None, find_camera_volumes)
            
            if camera_volumes:
                self.sd_card_path = camera_volumes[0]
                self.sd_input.value = str(self.sd_card_path)
                camera_type = detect_camera_type(self.sd_card_path)
                logger.info(f"Auto-detected SD card: {self.sd_card_path} ({camera_type.value})")
                
                if len(camera_volumes) > 1:
                    logger.warning(f"Multiple cameras detected: {[str(v) for v in camera_volumes]}")
            else:
                logger.info("No camera SD cards detected")
                
            self._update_button_states()
        except Exception as e:
            logger.error(f"Error detecting SD cards: {e}")

    async def select_project_folder(self, widget):
        """Select project folder manually."""
        try:
            folder = await self.main_window.select_folder_dialog(title="Select project folder")
            if folder:
                self.project_path = Path(folder)
                self.project_input.value = str(self.project_path)
                self.project_name = self.project_path.name
                logger.info(f"Selected project folder: {self.project_path}")
                self._update_button_states()
        except Exception as e:
            logger.error(f"Error selecting project folder: {e}")

    async def detect_project_folder(self, widget):
        """Auto-detect project folder from Acasis volume."""
        try:
            acasis_path = Path("/Volumes/Acasis")
            if acasis_path.exists():
                # Find most recent project directory
                project_dirs = sorted(
                    [d for d in acasis_path.iterdir() if d.is_dir() and not d.name.startswith('.')],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if project_dirs:
                    self.project_path = project_dirs[0]
                    self.project_input.value = str(self.project_path)
                    self.project_name = self.project_path.name
                    logger.info(f"Auto-detected project folder: {self.project_path}")
                else:
                    logger.info("No project folders found on Acasis")
            else:
                logger.info("Acasis volume not mounted")
                
            self._update_button_states()
        except Exception as e:
            logger.error(f"Error detecting project folder: {e}")

    async def select_proxy_folder(self, widget):
        """Select proxy folder manually."""
        try:
            folder = await self.main_window.select_folder_dialog(title="Select proxy folder")
            if folder:
                self.proxy_path = Path(folder)
                self.proxy_input.value = str(self.proxy_path)
                logger.info(f"Selected proxy folder: {self.proxy_path}")
                self._update_button_states()
        except Exception as e:
            logger.error(f"Error selecting proxy folder: {e}")

    async def select_outline_file(self, widget):
        """Select outline file for beat alignment."""
        try:
            file = await self.main_window.open_file_dialog(
                title="Select outline file",
                file_types=["md", "txt"]
            )
            if file:
                self.outline_path = Path(file)
                self.outline_input.value = str(self.outline_path)
                logger.info(f"Selected outline file: {self.outline_path}")
                self._update_button_states()
        except Exception as e:
            logger.error(f"Error selecting outline file: {e}")

    # === SETTINGS ===
    
    def open_settings(self, widget):
        """Open the settings window."""
        settings_window = SettingsWindow(self)
        settings_window.show()

    # === PHASE EXECUTION ===
    
    def _set_running(self, running: bool):
        """Set running state and update buttons."""
        self.is_running = running
        self._update_button_states()
        
    def _get_project_name(self) -> str:
        """Get or generate project name."""
        if self.project_name:
            return self.project_name
        if self.project_path:
            return self.project_path.name
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_proxy_dir(self) -> Path:
        """Get the proxy directory path."""
        project_name = self._get_project_name()
        return self.proxy_path / project_name / "proxy"

    async def run_copy_phase(self, widget):
        """Run Stage 1: Copy from SD card."""
        if not self.sd_card_path or not self.project_path:
            logger.error("SD card and project folder required for copy phase")
            return
        
        self._set_running(True)
        self.status_label.text = "Stage 1: Copying from SD card..."
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_copy():
                project_name = self._get_project_name()
                
                def progress_cb(name, i, total):
                    self.loop.call_soon_threadsafe(
                        self._update_progress, i, total, f"Copying {i}/{total}: {name}"
                    )
                
                session = ingest_volume(
                    self.sd_card_path,
                    self.project_path.parent,  # Parent of project folder (e.g. /Volumes/Acasis)
                    project_name,
                    progress_callback=progress_cb,
                )
                return session
            
            session = await loop.run_in_executor(None, do_copy)
            logger.info(f"Copy complete: {len(session.files)} files copied")
            self.status_label.text = f"Copy complete: {len(session.files)} files"
            
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            self.status_label.text = f"Copy failed: {e}"
        finally:
            self._set_running(False)

    async def run_proxy_phase(self, widget):
        """Run Stage 2: Generate Proxies."""
        if not self.project_path:
            logger.error("Project folder required for proxy generation")
            return
        
        self._set_running(True)
        self.status_label.text = "Stage 2: Generating proxies..."
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_proxies():
                # Find all video files in project folder
                video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mxf', '.MXF', '.mts', '.MTS', '.insv', '.INSV', '.insp', '.INSP'}
                source_files = []
                
                # Look in camera subdirectories
                for subdir in self.project_path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        for video_file in subdir.iterdir():
                            if video_file.is_file() and video_file.suffix in video_extensions:
                                source_files.append(video_file)
                
                if not source_files:
                    # Also check root of project folder
                    for video_file in self.project_path.iterdir():
                        if video_file.is_file() and video_file.suffix in video_extensions:
                            source_files.append(video_file)
                
                if not source_files:
                    raise ValueError(f"No video files found in {self.project_path}")
                
                proxy_dir = self._get_proxy_dir()
                logger.info(f"Generating proxies for {len(source_files)} files to {proxy_dir}")
                
                results = generate_proxies_batch(source_files, proxy_dir)
                successful = [r for r in results if r.success]
                return len(successful), len(results)
            
            success_count, total_count = await loop.run_in_executor(None, do_proxies)
            logger.info(f"Proxy generation complete: {success_count}/{total_count}")
            self.status_label.text = f"Proxies generated: {success_count}/{total_count}"
            
        except Exception as e:
            logger.error(f"Proxy generation failed: {e}")
            self.status_label.text = f"Proxy generation failed: {e}"
        finally:
            self._set_running(False)

    async def run_analysis_phase(self, widget):
        """Run Stage 3: AI Analysis."""
        proxy_dir = self._get_proxy_dir()
        
        # Find clips to analyze
        clips_to_analyze = []
        
        if proxy_dir.exists():
            for proxy_file in proxy_dir.glob("*.mp4"):
                clips_to_analyze.append((proxy_file, proxy_file))
        elif self.project_path:
            # Analyze source files directly
            video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mxf', '.MXF'}
            for subdir in self.project_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    for video_file in subdir.iterdir():
                        if video_file.is_file() and video_file.suffix in video_extensions:
                            clips_to_analyze.append((video_file, None))
        
        if not clips_to_analyze:
            logger.error("No clips found to analyze")
            return
        
        self._set_running(True)
        self.status_label.text = f"Stage 3: Analyzing {len(clips_to_analyze)} clips..."
        self.total_count = len(clips_to_analyze)
        self.progress_bar.max = self.total_count
        self.analysis_start_time = time.time()
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_analysis():
                analyses = analyze_clips_batch(
                    clips_to_analyze,
                    use_vlm=True,
                    model_name=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    provider_preferences=None,
                    max_workers=self.max_workers,
                )
                return analyses
            
            analyses = await loop.run_in_executor(None, do_analysis)
            
            # Update UI with results
            for i, analysis in enumerate(analyses):
                self.update_ui(i + 1, len(analyses), analysis)
            
            logger.info(f"Analysis complete: {len(analyses)} clips")
            self.status_label.text = f"Analysis complete: {len(analyses)} clips"
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.status_label.text = f"Analysis failed: {e}"
        finally:
            self._set_running(False)

    async def run_trim_phase(self, widget):
        """Run Stage 3.5: Trim Detection."""
        proxy_dir = self._get_proxy_dir()
        
        if not proxy_dir.exists():
            logger.error("Proxy directory not found")
            return
        
        self._set_running(True)
        self.status_label.text = "Stage 3.5: Detecting trims..."
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_trims():
                detect_trims_batch(
                    project_dir=proxy_dir,
                    model_name=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    provider_preferences=None,
                    max_workers=self.max_workers,
                )
            
            await loop.run_in_executor(None, do_trims)
            logger.info("Trim detection complete")
            self.status_label.text = "Trim detection complete"
            
        except Exception as e:
            logger.error(f"Trim detection failed: {e}")
            self.status_label.text = f"Trim detection failed: {e}"
        finally:
            self._set_running(False)

    async def run_beats_phase(self, widget):
        """Run Stage 4: Beat Alignment."""
        proxy_dir = self._get_proxy_dir()
        
        if not proxy_dir.exists():
            logger.error("Proxy directory not found")
            return
        
        if not self.outline_path or not self.outline_path.exists():
            logger.error("Outline file required for beat alignment")
            return
        
        self._set_running(True)
        self.status_label.text = "Stage 4: Aligning to beats..."
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_beats():
                align_beats(
                    project_dir=proxy_dir,
                    outline_path=self.outline_path,
                    model_name=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    provider_preferences=None,
                )
            
            await loop.run_in_executor(None, do_beats)
            logger.info("Beat alignment complete")
            self.status_label.text = "Beat alignment complete"
            
        except Exception as e:
            logger.error(f"Beat alignment failed: {e}")
            self.status_label.text = f"Beat alignment failed: {e}"
        finally:
            self._set_running(False)

    async def run_all_phases(self, widget):
        """Run all applicable phases."""
        has_sd = self.sd_card_path is not None and self.sd_card_path.exists()
        has_project = self.project_path is not None and self.project_path.exists()
        
        self._set_running(True)
        
        try:
            # Stage 1: Copy from SD if available
            if has_sd and has_project:
                await self.run_copy_phase(widget)
                self._set_running(True)  # Re-enable running state
            
            # Stage 2: Generate Proxies
            if has_project or has_sd:
                await self.run_proxy_phase(widget)
                self._set_running(True)
            
            # Stage 3: Analysis
            await self.run_analysis_phase(widget)
            self._set_running(True)
            
            # Stage 3.5: Trim Detection
            await self.run_trim_phase(widget)
            self._set_running(True)
            
            # Stage 4: Beat Alignment (if outline provided)
            if self.outline_path and self.outline_path.exists():
                await self.run_beats_phase(widget)
            
            self.status_label.text = "All phases complete!"
            logger.info("All phases complete!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.status_label.text = f"Pipeline failed: {e}"
        finally:
            self._set_running(False)

    # === UI HELPERS ===
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar and status from any thread."""
        self.progress_bar.max = total
        self.progress_bar.value = current
        self.status_label.text = message

    def _extract_thumbnail(self, video_path: Path, timestamp: float = 1.0) -> Optional[bytes]:
        """Extract a thumbnail from a video file."""
        import subprocess
        
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
        """Update UI elements."""
        self.processed_count = processed
        self.progress_bar.value = processed
        
        # Calculate ETA
        if self.analysis_start_time and processed > 0:
            elapsed = time.time() - self.analysis_start_time
            avg_speed = elapsed / processed
            remaining = total - processed
            eta_seconds = remaining * avg_speed
            
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}s"
            else:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                
            self.status_label.text = f"Processing: {processed}/{total} | ETA: {eta_str}"
        
        # Update main view if not in review mode
        if not self.is_review_mode and analysis:
            self.clip_label.text = analysis.source_path.name
            
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

        # Build details text
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
        
        # Show details panel
        if self.details_panel not in self.main_box.children:
            self.main_box.add(self.details_panel)

    def resume_live_view(self, widget):
        """Resume live view updates."""
        self.is_review_mode = False
        self.resume_button.enabled = False
        self.current_review_analysis = None
        self.preview_btn.enabled = False
        
        if self.details_panel in self.main_box.children:
            self.main_box.remove(self.details_panel)
            
        self.clip_label.text = "Resuming live view..."

    def add_recent_clip(self, analysis: ClipAnalysis):
        """Add a thumbnail to the recent clips strip."""
        thumb_box = toga.Box(style=Pack(direction=COLUMN, width=140, margin=5))
        
        try:
            video_path = analysis.proxy_path or analysis.source_path
            timestamp = analysis.thumbnail_timestamp_sec or 1.0
            thumb_bytes = self._extract_thumbnail(video_path, timestamp)
            
            if thumb_bytes:
                toga_img = toga.Image(src=thumb_bytes)
                image_view = toga.ImageView(image=toga_img, style=Pack(height=80, width=120))
                thumb_box.add(image_view)
            
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
        
        clip_name = analysis.clip_name or analysis.source_path.stem
        if len(clip_name) > 18:
            clip_name = clip_name[:16] + ".."
        name_label = toga.Label(clip_name, style=Pack(text_align=CENTER, font_size=10))
        thumb_box.add(name_label)
        
        duration_label = toga.Label(f"{analysis.duration_seconds:.1f}s", style=Pack(text_align=CENTER, font_size=9))
        thumb_box.add(duration_label)
        
        self.recent_box.insert(0, thumb_box)


def main(
    sd_card_path: Optional[Path] = None,
    project_path: Optional[Path] = None,
    proxy_path: Optional[Path] = None,
    model: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_workers: int = 1,
):
    """Create and return the TVAS Status App."""
    return TvasStatusApp(
        sd_card_path=sd_card_path,
        project_path=project_path,
        proxy_path=proxy_path,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_workers=max_workers,
    )
