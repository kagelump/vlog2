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
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from PIL import Image, ImageOps
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from shared import DEFAULT_VLM_MODEL, load_prompt, set_prompt_override, get_openrouter_api_key
from shared.vlm_client import CostTracker
from shared.paths import detect_archival_root, find_latest_project
from shared.ffmpeg_utils import extract_thumbnail
from tvas.analysis import ClipAnalysis, analyze_clips_batch
from tvas.trim import detect_trims_batch
from tvas.beats import align_beats
from tvas.ingestion import CameraType, detect_camera_type, ingest_volume, get_video_files
from tvas.watcher import find_camera_volumes, is_camera_volume
from shared.proxy import generate_proxies_batch

# === STYLE CONSTANTS ===
# Customize these constants to adjust the UI appearance throughout the application
STYLES = {
    # Button styles
    'button_phase': Pack(margin=(2, 3), height=24),
    'button_phase_bold': Pack(margin=(2, 3), height=24, font_weight='bold'),
    'button_settings': Pack(margin=(2, 3), height=24),
    'button_small': Pack(height=28, width=120, font_size=10, margin_top=2),
    'button_action': Pack(margin=(0, 5)),
    
    # Layout styles
    'layout_column': Pack(direction=COLUMN),
    'layout_row': Pack(direction=ROW),
    'layout_column_flex': Pack(direction=COLUMN, flex=1),
    'layout_row_flex': Pack(direction=ROW, flex=1),
    'layout_column_margin': Pack(direction=COLUMN, margin=5),
    'layout_row_margin': Pack(direction=ROW, margin=5),
    'layout_row_section': Pack(direction=ROW, margin=5),
    'layout_column_section': Pack(direction=COLUMN, margin=5, flex=1),
    
    # Label styles
    'label_section_title': Pack(font_weight='bold', margin=(10, 10, 5, 10)),
    'label_subsection': Pack(font_weight='bold', margin=(5, 5)),
    'label_text': Pack(margin=5),
    'label_text_centered': Pack(margin=5, text_align=CENTER),
    'label_text_monospace': Pack(margin=(0, 10), font_family="monospace", font_size=10, flex=1),
    
    # Input styles
    'input_text': Pack(flex=1, margin=(0, 5)),
    'input_readonly': Pack(flex=1, margin=(0, 5)),
    
    # Container styles
    'container_details': Pack(direction=COLUMN, width=350, margin=10),
    'container_main': Pack(direction=ROW, flex=1, margin=10),
    'container_footer': Pack(direction=COLUMN),
    'container_recent': Pack(direction=ROW, margin=10),
    'container_clip_thumb': Pack(direction=COLUMN, width=140, margin=5),
    'container_clip_image': Pack(width=120, height=67, align_items=CENTER),
    
    # Progress styles
    'progress_bar': Pack(margin=(0, 10), flex=1),
    
    # Divider styles
    'divider': Pack(height=1, background_color='#CCCCCC', margin=(10, 5)),
}

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
        prompt_files = ["video_describe.txt", "video_trim.txt", "beat_alignment.txt"]
        
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
        self.total_processing_time = 0.0
        self.items_with_timing = 0
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
        
        # Stage 4: Beat Alignment - needs analyses and outline
        self.beats_btn.enabled = has_analyses and has_outline and not running
        
        # Stage 5: Trim Detection - needs analysis JSON files
        self.trim_btn.enabled = has_analyses and not running
        
        # Run Ingestion (1-3) - needs at least project folder or SD card
        self.run_ingest_btn.enabled = (has_sd or has_project) and not running
        
        # Run Post-Processing (4-5) - needs analyses and outline
        self.run_post_btn.enabled = has_analyses and has_outline and not running

    def startup(self):
        """Construct and show the Toga application."""
        
        # === PHASE BUTTONS SECTION (LEFT SIDE) ===
        phase_section_label = toga.Label("Pipeline Phases", style=STYLES['label_section_title'])
        
        # Individual phase buttons
        self.copy_btn = toga.Button(
            "1. Copy from SD",
            on_press=self.run_copy_phase,
            enabled=False,
            style=STYLES['button_phase']
        )
        self.proxy_btn = toga.Button(
            "2. Generate Proxies",
            on_press=self.run_proxy_phase,
            enabled=False,
            style=STYLES['button_phase']
        )
        self.analyze_btn = toga.Button(
            "3. AI Analysis",
            on_press=self.run_analysis_phase,
            enabled=False,
            style=STYLES['button_phase']
        )
        self.beats_btn = toga.Button(
            "4. Beat Alignment",
            on_press=self.run_beats_phase,
            enabled=False,
            style=STYLES['button_phase']
        )
        self.trim_btn = toga.Button(
            "5. Trim Detection",
            on_press=self.run_trim_phase,
            enabled=False,
            style=STYLES['button_phase']
        )
        
        # Divider line
        divider1 = toga.Box(style=STYLES['divider'])
        
        # Run pipeline buttons - more prominent
        self.run_ingest_btn = toga.Button(
            "▶ Run Ingestion (1-3)",
            on_press=self.run_ingestion_pipeline,
            enabled=False,
            style=STYLES['button_phase_bold']
        )
        self.run_post_btn = toga.Button(
            "▶ Run Post (4-5)",
            on_press=self.run_post_pipeline,
            enabled=False,
            style=STYLES['button_phase_bold']
        )
        
        # Settings button
        self.settings_btn = toga.Button(
            "Settings",
            on_press=self.open_settings,
            style=STYLES['button_settings']
        )
        
        # Column 1: Individual Manual Phases
        phase_col1 = toga.Box(
            children=[
                toga.Label("Manual Steps", style=STYLES['label_subsection']),
                self.copy_btn,
                self.proxy_btn,
                self.analyze_btn,
                self.beats_btn,
                self.trim_btn,
            ],
            style=STYLES['layout_column_flex']
        )

        # Column 2: Automation Pipelines & Settings
        phase_col2 = toga.Box(
            children=[
                toga.Label("Automation", style=STYLES['label_subsection']),
                self.run_ingest_btn,
                self.run_post_btn,
                toga.Box(style=Pack(flex=1)),  # Spacer
                self.settings_btn
            ],
            style=Pack(direction=COLUMN, flex=1, margin_left=10)  # margin_left not in STYLES dict
        )

        phase_box = toga.Box(
            children=[
                phase_section_label,
                toga.Box(
                    children=[phase_col1, phase_col2],
                    style=Pack(direction=ROW)
                )
            ],
            style=Pack(direction=COLUMN, margin=5, width=400)
        )
        
        # === PATH SELECTION SECTION (RIGHT SIDE) ===
        path_section_label = toga.Label("Paths", style=STYLES['label_section_title'])
        
        # SD Card Volume
        self.sd_input = toga.TextInput(
            readonly=True,
            placeholder="Auto-detect or select SD card...",
            style=STYLES['input_readonly']
        )
        if self.sd_card_path:
            self.sd_input.value = str(self.sd_card_path)
            
        self.sd_browse_btn = toga.Button("Browse...", on_press=self.select_sd_card, style=STYLES['button_action'])
        self.sd_detect_btn = toga.Button("Detect", on_press=self.detect_sd_card, style=STYLES['button_action'])
        
        sd_row = toga.Box(
            children=[
                toga.Label("SD Card:", style=Pack(margin=(5, 5), width=100)),
                self.sd_input,
                self.sd_browse_btn,
                self.sd_detect_btn,
            ],
            style=STYLES['layout_row_section']
        )
        
        # Project Folder (archival storage)
        self.project_input = toga.TextInput(
            readonly=True,
            placeholder="Project folder (e.g. /Volumes/Acasis/project_name)...",
            style=STYLES['input_readonly']
        )
        if self.project_path:
            self.project_input.value = str(self.project_path)
            self.project_name = self.project_path.name

        self.project_browse_btn = toga.Button("Browse...", on_press=self.select_project_folder, style=STYLES['button_action'])
        self.project_detect_btn = toga.Button("Detect", on_press=self.detect_project_folder, style=STYLES['button_action'])
        
        project_row = toga.Box(
            children=[
                toga.Label("Project:", style=Pack(margin=(5, 5), width=100)),
                self.project_input,
                self.project_browse_btn,
                self.project_detect_btn,
            ],
            style=STYLES['layout_row_section']
        )
        
        # Proxy Folder
        self.proxy_input = toga.TextInput(
            readonly=True,
            placeholder="Proxy folder (default: ~/Movies/Vlog)...",
            style=STYLES['input_readonly']
        )
        if self.proxy_path:
            self.proxy_input.value = str(self.proxy_path)
        self.proxy_browse_btn = toga.Button("Browse...", on_press=self.select_proxy_folder, style=STYLES['button_action'])
        # Spacer button to align with rows that have two buttons
        proxy_spacer = toga.Box(style=Pack(width=65, margin=(0, 5)))
        
        proxy_row = toga.Box(
            children=[
                toga.Label("Proxy:", style=Pack(margin=(5, 5), width=100)),
                self.proxy_input,
                self.proxy_browse_btn,
                proxy_spacer,
            ],
            style=STYLES['layout_row_section']
        )
        
        # Outline file (for beat alignment)
        self.outline_input = toga.TextInput(
            readonly=True,
            placeholder="Optional: outline.md for beat alignment...",
            style=STYLES['input_readonly']
        )
        self.outline_browse_btn = toga.Button("Browse...", on_press=self.select_outline_file, style=STYLES['button_action'])
        # Spacer to align with rows that have two buttons
        outline_spacer = toga.Box(style=Pack(width=65, margin=(0, 5)))
        
        outline_row = toga.Box(
            children=[
                toga.Label("Outline:", style=Pack(margin=(5, 5), width=100)),
                self.outline_input,
                self.outline_browse_btn,
                outline_spacer,
            ],
            style=STYLES['layout_row_section']
        )
        
        path_box = toga.Box(
            children=[path_section_label, sd_row, project_row, proxy_row, outline_row],
            style=STYLES['layout_column_section']
        )
        
        # Combine phase buttons and paths horizontally
        top_section = toga.Box(
            children=[phase_box, path_box],
            style=STYLES['layout_row_section']
        )
        
        # === PROGRESS SECTION ===
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=STYLES['progress_bar'])
        
        self.mode_label = toga.Label("", style=Pack(margin=(5, 5), font_weight='bold'))
        self.update_mode_label()
        
        self.status_label = toga.Label("Ready", style=Pack(margin=(5, 5), flex=1))
        self.cost_label = toga.Label("", style=Pack(margin=(5, 5), color="#888888"))
        
        status_row = toga.Box(
            children=[self.mode_label, self.status_label, self.cost_label],
            style=STYLES['layout_row']
        )

        self.log_label = toga.Label(
            "Configure paths and click a phase button to start", 
            style=STYLES['label_text_monospace']
        )
        
        self.stop_button = toga.Button(
            "⏹ Stop",
            on_press=self.stop_processing,
            enabled=False,
            style=Pack(margin=(0, 5), width=70)
        )
        
        self.resume_button = toga.Button(
            "Resume Live View",
            on_press=self.resume_live_view,
            enabled=False,
            style=STYLES['button_action']
        )

        log_row = toga.Box(
            children=[self.log_label, self.stop_button, self.resume_button],
            style=Pack(direction=ROW, align_items=CENTER)
        )
        
        progress_box = toga.Box(
            children=[status_row, self.progress_bar, log_row],
            style=Pack(direction=COLUMN, margin=10)
        )

        # === MAIN CONTENT: Clip View & Details ===
        self.image_view = toga.ImageView(style=Pack(flex=1))
        self.clip_label = toga.Label("No clip loaded", style=STYLES['label_text_centered'])
        
        self.image_area = toga.Box(
            children=[self.image_view, self.clip_label],
            style=STYLES['layout_column_flex']
        )

        # Details Panel
        self.details_label = toga.Label("Details", style=Pack(font_weight='bold', margin_bottom=5))
        self.details_content = toga.MultilineTextInput(readonly=True, style=Pack(flex=1))
        self.preview_btn = toga.Button("Preview Frame", on_press=self.open_preview, enabled=False, style=Pack(margin_top=5))
        
        self.details_panel = toga.Box(
            children=[self.details_label, self.details_content, self.preview_btn],
            style=STYLES['container_details']
        )
        
        main_box = toga.Box(
            children=[self.image_area],
            style=STYLES['container_main']
        )
        self.main_box = main_box

        # === FOOTER: Recent Clips ===
        self.recent_box = toga.Box(style=STYLES['container_recent'])
        
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
            children=[top_section, progress_box, main_box, footer_container],
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
        
        # Only auto-detect if not already set (e.g. from CLI args)
        if not self.sd_card_path:
            await self.detect_sd_card(None)
            
        if not self.project_path:
            await self.detect_project_folder(None)
            
        self._update_button_states()

    # === PATH SELECTION HANDLERS ===
    
    async def select_sd_card(self, widget):
        """Select SD card folder manually."""
        try:
            folder = await self.main_window.dialog(toga.SelectFolderDialog(title="Select SD card volume"))
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
            folder = await self.main_window.dialog(toga.SelectFolderDialog(title="Select project folder"))
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
            acasis_path = await asyncio.get_running_loop().run_in_executor(None, detect_archival_root, None)
            
            if acasis_path and acasis_path.exists():
                # Find most recent project directory
                latest_project = await asyncio.get_running_loop().run_in_executor(None, find_latest_project, acasis_path)
                
                if latest_project:
                    self.project_path = latest_project
                    self.project_input.value = str(self.project_path)
                    self.project_name = self.project_path.name
                    logger.info(f"Auto-detected project folder: {self.project_path}")
                else:
                    logger.info("No project folders found on Acasis")
            else:
                logger.info("Acasis volume not mounted")
        except Exception as e:
            logger.error(f"Error detecting project folder: {e}")
                
            self._update_button_states()
        except Exception as e:
            logger.error(f"Error detecting project folder: {e}")

    async def select_proxy_folder(self, widget):
        """Select proxy folder manually."""
        try:
            folder = await self.main_window.dialog(toga.SelectFolderDialog(title="Select proxy folder"))
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
            file = await self.main_window.dialog(toga.OpenFileDialog(
                title="Select outline file",
                file_types=["md", "txt"]
            ))
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
        self.stop_button.enabled = running
        if not running:
            self.stop_event.clear()  # Reset stop event when not running
        self._update_button_states()
    
    async def stop_processing(self, widget):
        """Stop the current processing operation."""
        if self.is_running:
            logger.info("Stopping processing...")
            self.stop_event.set()
            self.status_label.text = "Stopping..."
            self.stop_button.enabled = False
        
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
                    self.main_window.app.loop.call_soon_threadsafe(
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
                            if video_file.is_file() and video_file.suffix in video_extensions and not video_file.name.startswith('.'):
                                source_files.append(video_file)
                
                if not source_files:
                    # Also check root of project folder
                    for video_file in self.project_path.iterdir():
                        if video_file.is_file() and video_file.suffix in video_extensions and not video_file.name.startswith('.'):
                            source_files.append(video_file)
                
                if not source_files:
                    raise ValueError(f"No video files found in {self.project_path}")
                
                proxy_dir = self._get_proxy_dir()
                logger.info(f"Generating proxies for {len(source_files)} files to {proxy_dir}")
                
                def on_progress(current, total, result):
                    msg = f"Proxies: {current}/{total} - {result.source_path.name}"
                    self.main_window.app.loop.call_soon_threadsafe(
                        self._update_progress, current, total, msg
                    )
                
                results = generate_proxies_batch(
                    source_files, 
                    proxy_dir,
                    progress_callback=on_progress
                )
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
                if not proxy_file.name.startswith('.'):
                    clips_to_analyze.append((proxy_file, proxy_file))
        elif self.project_path:
            # Analyze source files directly
            video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mxf', '.MXF'}
            for subdir in self.project_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    for video_file in subdir.iterdir():
                        if video_file.is_file() and video_file.suffix in video_extensions and not video_file.name.startswith('.'):
                            clips_to_analyze.append((video_file, None))
        
        if not clips_to_analyze:
            logger.error("No clips found to analyze")
            return
        
        self._set_running(True)
        self.status_label.text = f"Stage 3: Analyzing {len(clips_to_analyze)} clips..."
        self.total_count = len(clips_to_analyze)
        self.progress_bar.max = self.total_count
        self.analysis_start_time = time.time()
        self.total_processing_time = 0.0
        self.items_with_timing = 0
        
        try:
            loop = asyncio.get_running_loop()
            
            def do_analysis():
                def on_progress(current, total, result):
                    self.main_window.app.loop.call_soon_threadsafe(
                        self.update_ui, current, total, result
                    )
                
                analyses = analyze_clips_batch(
                    clips_to_analyze,
                    use_vlm=True,
                    model_name=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    provider_preferences=None,
                    max_workers=self.max_workers,
                    progress_callback=on_progress
                )
                return analyses
            
            analyses = await loop.run_in_executor(None, do_analysis)
            
            # Display cumulative cost
            total_cost = CostTracker.get_total()
            cost_msg = f" | Cost: ${total_cost:.4f}" if total_cost > 0 else ""
            
            logger.info(f"Analysis complete: {len(analyses)} clips{cost_msg}")
            self.status_label.text = f"Analysis complete: {len(analyses)} clips{cost_msg}"
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.status_label.text = f"Analysis failed: {e}"
        finally:
            self._set_running(False)

    async def run_trim_phase(self, widget):
        """Run Stage 5: Trim Detection."""
        proxy_dir = self._get_proxy_dir()
        
        if not proxy_dir.exists():
            logger.error("Proxy directory not found")
            return
        
        self._set_running(True)
        self.status_label.text = "Stage 5: Detecting trims..."
        
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
            
            # Display cumulative cost
            total_cost = CostTracker.get_total()
            cost_msg = f" | Cost: ${total_cost:.4f}" if total_cost > 0 else ""
            
            logger.info(f"Trim detection complete{cost_msg}")
            self.status_label.text = f"Trim detection complete{cost_msg}"
            
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
            
            # Display cumulative cost
            total_cost = CostTracker.get_total()
            cost_msg = f" | Cost: ${total_cost:.4f}" if total_cost > 0 else ""
            
            logger.info(f"Beat alignment complete{cost_msg}")
            self.status_label.text = f"Beat alignment complete{cost_msg}"
            
        except Exception as e:
            logger.error(f"Beat alignment failed: {e}")
            self.status_label.text = f"Beat alignment failed: {e}"
        finally:
            self._set_running(False)

    async def run_ingestion_pipeline(self, widget):
        """Run Ingestion Pipeline (Phase 1-3)."""
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
            
            self.status_label.text = "Ingestion pipeline complete!"
            logger.info("Ingestion pipeline complete!")
            
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            self.status_label.text = f"Ingestion pipeline failed: {e}"
        finally:
            self._set_running(False)

    async def run_post_pipeline(self, widget):
        """Run Post-Processing Pipeline (Phase 4-5)."""
        if not self.outline_path or not self.outline_path.exists():
            self.main_window.info_dialog("Outline Required", "Beat alignment requires an outline.md file.")
            return

        self._set_running(True)
        
        try:
            # Stage 4: Beat Alignment
            await self.run_beats_phase(widget)
            self._set_running(True)
            
            # Stage 5: Trim Detection
            await self.run_trim_phase(widget)
            
            self.status_label.text = "Post-processing pipeline complete!"
            logger.info("Post-processing pipeline complete!")
            
        except Exception as e:
            logger.error(f"Post-processing pipeline failed: {e}")
            self.status_label.text = f"Post-processing pipeline failed: {e}"
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
        """Update UI elements.
        
        Note: Preview image loading is deferred during parallel processing to avoid
        blocking the main thread with ffmpeg calls. Users can click 'View' to see
        full previews on demand.
        """
        self.processed_count = processed
        self.progress_bar.value = processed
        
        # Track actual processing time for accurate ETA
        if analysis and analysis.analysis_duration > 0:
            self.total_processing_time += analysis.analysis_duration
            self.items_with_timing += 1
        
        # Calculate ETA based on actual processing times
        if self.analysis_start_time and processed > 0:
            # Use actual measured processing times if available
            if self.items_with_timing > 0:
                # Average processing time per item, adjusted for parallel workers
                avg_processing_time = self.total_processing_time / self.items_with_timing
                # Wall-clock time per item = processing time / number of workers
                avg_speed = avg_processing_time / max(1, self.max_workers)
            else:
                # Fallback to wall-clock average if no timing data yet
                elapsed = time.time() - self.analysis_start_time
                avg_speed = elapsed / processed
            
            remaining = total - processed
            eta_seconds = max(0, remaining * avg_speed)
            
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}s"
            else:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                
            self.status_label.text = f"Processing: {processed}/{total} | ETA: {eta_str}"
        
        # Update cumulative cost display if in API mode
        total_cost = CostTracker.get_total()
        if total_cost > 0:
            self.cost_label.text = f"💰 ${total_cost:.4f}"
        
        # Update clip name but skip heavy preview loading during parallel processing
        # to avoid blocking the main thread and causing UI thread crashes
        if not self.is_review_mode and analysis:
            self.clip_label.text = analysis.source_path.name
            # Skip live preview image loading - users can click 'View' for full preview

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

    def _load_thumbnail_background(self, analysis: ClipAnalysis, container: toga.Box, spinner: toga.ActivityIndicator):
        """Background task to load/generate thumbnail."""
        try:
            # Determine cache path
            cache_dir = Path.home() / ".tvas" / "thumbnails"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename based on path hash
            path_hash = hashlib.md5(str(analysis.source_path).encode()).hexdigest()
            thumb_path = cache_dir / f"{path_hash}.jpg"
            
            video_path = analysis.proxy_path or analysis.source_path
            
            success = False
            if thumb_path.exists():
                success = True
            elif video_path and video_path.exists():
                # Extract if not exists
                timestamp = analysis.thumbnail_timestamp_sec or 1.0
                success = extract_thumbnail(video_path, thumb_path, timestamp)
            
            def update_ui():
                try:
                    # Remove spinner
                    if spinner in container.children:
                        container.remove(spinner)
                    
                    if success:
                        # Create and add image view
                        image_view = toga.ImageView(
                            image=toga.Image(thumb_path),
                            style=Pack(width=120, height=67) # ~16:9
                        )
                        container.add(image_view)
                    else:
                        # Fallback showing error or placeholder
                        lbl = toga.Label("No Preview", style=Pack(width=120, height=67, text_align=CENTER))
                        container.add(lbl)
                except Exception as e:
                    logger.error(f"Failed to update thumbnail UI: {e}")

            self.loop.call_soon_threadsafe(update_ui)
                
        except Exception as e:
            logger.error(f"Error loading thumbnail: {e}")
            # Ensure spinner is removed on error
            def clean_error():
                if spinner in container.children:
                    container.remove(spinner)
                lbl = toga.Label("Error", style=Pack(width=120, height=67, text_align=CENTER))
                container.add(lbl)
            self.loop.call_soon_threadsafe(clean_error)

    def add_recent_clip(self, analysis: ClipAnalysis):
        """Add a thumbnail to the recent clips strip."""
        # Limit the number of recent clips to prevent unbounded UI widget growth
        # Remove oldest clips if we have too many (widgets must be removed on main thread)
        max_recent = 20
        while len(self.recent_box.children) >= max_recent:
            try:
                oldest = self.recent_box.children[-1]
                self.recent_box.remove(oldest)
            except Exception:
                break
        
        thumb_box = toga.Box(style=STYLES['container_clip_thumb'])
        
        # Container for the image area
        img_container = toga.Box(style=STYLES['container_clip_image'])
        thumb_box.add(img_container)
        
        # Start with spinner
        spinner = toga.ActivityIndicator(style=Pack(margin_top=20))
        spinner.start()
        img_container.add(spinner)

        # Add a small Details button below
        view_btn = toga.Button(
            "Details", 
            on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
            style=STYLES['button_small']
        )
        thumb_box.add(view_btn)
        
        clip_name = analysis.clip_name or analysis.source_path.stem
        if len(clip_name) > 18:
            clip_name = clip_name[:16] + ".."
        name_label = toga.Label(clip_name, style=Pack(text_align=CENTER, font_size=10))
        thumb_box.add(name_label)
        
        duration_label = toga.Label(f"{analysis.duration_seconds:.1f}s", style=Pack(text_align=CENTER, font_size=9))
        thumb_box.add(duration_label)
        
        self.recent_box.insert(0, thumb_box)
        
        # Store reference to prevent premature GC
        self.recent_clips.append(analysis)
        
        # Start background loading
        threading.Thread(
            target=self._load_thumbnail_background, 
            args=(analysis, img_container, spinner),
            daemon=True
        ).start()


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
