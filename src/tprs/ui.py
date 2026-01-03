"""TPRS Status GUI

A non-interactive GUI for monitoring the Travel Photo Rating System progress.
"""

import asyncio
import logging
import threading
import functools
import tempfile
import io
import gc
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageOps
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from tprs.tprs import PhotoAnalysis, process_photos_batch, find_jpeg_photos, load_analysis_from_xmp
from shared import DEFAULT_VLM_MODEL, load_prompt, set_prompt_override

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

        general_box = toga.Box(
            children=[
                toga.Box(children=[toga.Label("Model:", style=Pack(width=100)), self.model_input], style=Pack(direction=ROW, margin=5)),
                toga.Box(children=[toga.Label("API Base:", style=Pack(width=100)), self.api_base_input], style=Pack(direction=ROW, margin=5)),
                toga.Box(children=[toga.Label("API Key:", style=Pack(width=100)), self.api_key_input], style=Pack(direction=ROW, margin=5)),
            ],
            style=Pack(direction=COLUMN, margin=10)
        )

        # Prompts
        self.prompt_inputs = {}
        prompt_files = [
            "photo_analysis.txt",
            "subject_sharpness.txt",
            "burst_similarity.txt",
            "best_in_burst.txt"
        ]
        
        prompt_container = toga.OptionContainer(style=Pack(flex=1))
        
        for pf in prompt_files:
            try:
                content = load_prompt(pf)
            except:
                content = ""
            text_input = toga.MultilineTextInput(value=content, style=Pack(flex=1, font_family="monospace"))
            self.prompt_inputs[pf] = text_input
            # OptionContainer expects content to be a list of OptionItem
            tab_content = toga.Box(children=[text_input], style=Pack(flex=1, margin=5))
            prompt_container.content.append(toga.OptionItem(pf.replace(".txt", ""), tab_content))

        # Buttons
        save_btn = toga.Button("Apply", on_press=self.save_settings, style=Pack(margin=5))
        close_btn = toga.Button("Close", on_press=self.close_window, style=Pack(margin=5))
        
        # Use spacer to align right
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
        
        for pf, input_widget in self.prompt_inputs.items():
            set_prompt_override(pf, input_widget.value)
            
        self.app_instance.update_mode_label()
        self.app_instance.main_window.info_dialog("Settings", "Settings applied for this session.")
        self.close()

    def close_window(self, widget):
        self.close()


class FocusCheckWindow(toga.Window):
    def __init__(self, app_instance, analysis):
        super().__init__(title=f"Focus Check: {analysis.photo_path.name}", size=(800, 800))
        self.app_instance = app_instance
        self.analysis = analysis
        self.init_ui()

    def init_ui(self):
        # We'll create the image view after loading to know the size
        image_view = None
        
        try:
            with Image.open(self.analysis.photo_path) as img:
                img = ImageOps.exif_transpose(img)
                if self.analysis.primary_subject_bounding_box:
                    width, height = img.size
                    xmin, ymin, xmax, ymax = self.analysis.primary_subject_bounding_box
                    
                    left = int((xmin / 1000) * width)
                    top = int((ymin / 1000) * height)
                    right = int((xmax / 1000) * width)
                    bottom = int((ymax / 1000) * height)
                    
                    # Ensure valid crop
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width, right)
                    bottom = min(height, bottom)
                    
                    # Add some padding (10%)
                    pad_x = int((right - left) * 0.1)
                    pad_y = int((bottom - top) * 0.1)
                    
                    left = max(0, left - pad_x)
                    top = max(0, top - pad_y)
                    right = min(width, right + pad_x)
                    bottom = min(height, bottom + pad_y)
                    
                    if right > left and bottom > top:
                        img = img.crop((left, top, right, bottom))
                
                # Save to temp file for Toga
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img.save(tf.name, quality=95)
                tf.close()
                
                # Create image view with explicit size for 2:1 display (200% zoom)
                # We set the style width/height to double the image dimensions
                img_w, img_h = img.size
                image_view = toga.ImageView(
                    image=toga.Image(tf.name),
                    style=Pack(width=img_w * 2, height=img_h * 2)
                )
        except Exception as e:
            logger.error(f"Failed to load focus check image: {e}")
            
        if image_view is None:
            image_view = toga.ImageView(style=Pack(flex=1))

        # Wrap in ScrollContainer to allow panning around the full resolution image
        scroll_container = toga.ScrollContainer(horizontal=True, vertical=True, style=Pack(flex=1))
        scroll_container.content = image_view
        
        self.content = scroll_container


class TprsStatusApp(toga.App):
    def __init__(self, directory: Optional[Path] = None, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL, api_base: Optional[str] = None, api_key: str = "lm-studio"):
        super().__init__("TPRS Status", "com.tvas.tprs_status")
        self.directory = directory
        self.output_dir = output_dir
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.processed_count = 0
        self.total_count = 0
        self.recent_photos = []  # List of (path, rating)
        self.is_running = False
        self.is_review_mode = False  # Track if user is viewing details
        self.stop_event = threading.Event()
        self.on_exit = self.exit_handler
        self.analysis_start_time = None
        self.initial_processed_count = None

    def exit_handler(self, app):
        """Handle app exit."""
        if self.is_running:
            logger.info("Stopping analysis...")
            self.stop_event.set()
            # Give it a moment? No, just return True and let it close.
            # The background thread will stop eventually.
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

        # --- Main: Current Photo & Details ---
        
        # Use a flexible height and width to allow the image to scale properly
        self.image_view = toga.ImageView(style=Pack(flex=1))
        self.image_view_2 = toga.ImageView(style=Pack(flex=1))
        
        self.images_container = toga.Box(
            children=[self.image_view],
            style=Pack(direction=ROW, flex=1)
        )
        
        self.photo_label = toga.Label("No photo loaded", style=Pack(margin=5, text_align=CENTER))
        
        self.image_area = toga.Box(
            children=[self.images_container, self.photo_label],
            style=Pack(direction=COLUMN, flex=1)
        )

        # Details Panel (Hidden by default or empty)
        self.details_label = toga.Label("Details", style=Pack(font_weight='bold', margin_bottom=5))
        self.details_content = toga.MultilineTextInput(readonly=True, style=Pack(flex=1))
        self.focus_check_btn = toga.Button("Focus Check (2x)", on_press=self.open_focus_check, enabled=False, style=Pack(margin_top=5))
        
        self.details_panel = toga.Box(
            children=[self.details_label, self.details_content, self.focus_check_btn],
            style=Pack(direction=COLUMN, width=300, margin=10)
        )
        # Initially hide details panel by removing it or setting width 0? 
        # Toga doesn't support hiding easily, so we'll just not add it to main_box initially?
        # Or we can just keep it there but empty.
        
        main_box = toga.Box(
            children=[self.image_area], # details_panel added dynamically
            style=Pack(direction=ROW, flex=1, margin=10)
        )
        self.main_box = main_box # Keep reference

        # --- Footer: Recent Photos ---
        self.recent_box = toga.Box(style=Pack(direction=ROW, margin=10))
        
        self.recent_scroll = toga.ScrollContainer(
            horizontal=True,
            vertical=False,
            style=Pack(height=150, flex=1)
        )
        self.recent_scroll.content = self.recent_box
        
        footer_container = toga.Box(
            children=[toga.Label("Recent Processed", style=Pack(margin=5)), self.recent_scroll],
            style=Pack(direction=COLUMN)
        )

        # --- Main Layout ---
        self.main_window = toga.MainWindow(title=self.formal_name, size=(1000, 800))
        self.main_window.content = toga.Box(
            children=[folder_row, header_box, main_box, footer_container],
            style=Pack(direction=COLUMN)
        )
        
        # Setup Logging
        handler = GuiLogHandler(self)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.main_window.show()

        # Attempt to maximize window to fill screen
        try:
            # Try standard Toga API for maximization
            self.main_window.state = toga.WindowState.MAXIMIZED
        except AttributeError:
            # Fallback: Set to screen size manually
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
        await asyncio.sleep(0.5) # Give UI time to appear
        await self.load_existing_xmps()
        await self.start_analysis(self.start_button)
    
    def open_settings(self, widget):
        """Open the settings window."""
        settings_window = SettingsWindow(self)
        settings_window.show()

    async def load_existing_xmps(self):
        """Load existing XMP files for photos in the directory."""
        if not self.directory:
            return

        self.status_label.text = "Checking for existing XMP files..."
        
        # Run in executor to avoid blocking UI during file I/O
        loop = asyncio.get_running_loop()
        
        def _load():
            photos = find_jpeg_photos(self.directory)
            loaded = []
            for photo_path in photos:
                if self.output_dir:
                    xmp_path = self.output_dir / f"{photo_path.stem}.xmp"
                else:
                    xmp_path = photo_path.with_suffix(".xmp")
                
                if xmp_path.exists():
                    try:
                        analysis = load_analysis_from_xmp(xmp_path, photo_path)
                        loaded.append(analysis)
                    except Exception as e:
                        logger.warning(f"Failed to load XMP for {photo_path}: {e}")
            return loaded

        existing_analyses = await loop.run_in_executor(None, _load)
        
        if existing_analyses:
            logger.info(f"Loaded {len(existing_analyses)} existing XMP files.  Press 'Start Analysis' to begin.")
            self.status_label.text = f"Loaded {len(existing_analyses)} existing XMP files.  Press 'Start Analysis' to begin."
            
            # Add to recent strip
            # We add them in order, so the last one in the list (lexicographically last)
            # ends up at the start of the strip (most recent).
            for analysis in existing_analyses:
                self.add_recent_photo(analysis)
                # Yield to event loop to keep UI responsive if many items
                await asyncio.sleep(0.01)
        else:
            self.status_label.text = "No existing XMP files found."

    async def select_folder(self, widget):
        """Handle folder selection."""
        try:
            # Use Toga's folder selection dialog
            folder = await self.main_window.select_folder_dialog(
                title="Select folder to scan for photos"
            )
            
            if folder:
                self.directory = Path(folder)
                self.folder_input.value = str(self.directory)
                self.start_button.enabled = True
                self.status_label.text = "Folder selected. Click Start Analysis to begin."
                logger.info(f"Selected folder: {self.directory}")
                
                # Load existing XMPs
                await self.load_existing_xmps()
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
        
        # Disable the start button during analysis
        self.start_button.enabled = False
        self.folder_button.enabled = False
        self.stop_event.clear()
        
        # Force garbage collection on main thread to clean up any UI objects
        # that might otherwise be collected in the background thread, causing a crash.
        gc.collect()
        
        try:
            # Start processing
            await self.run_analysis(widget)
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            self.status_label.text = f"Analysis failed: {e}"
            # Ensure buttons are re-enabled even on unexpected errors
            self.start_button.enabled = True
            self.folder_button.enabled = True
            self.is_running = False

    async def run_analysis(self, widget):
        """Run the analysis in a background thread."""
        self.is_running = True
        self.status_label.text = f"Scanning {self.directory}..."
        self.analysis_start_time = time.time()
        self.initial_processed_count = None
        
        # Find photos first (fast enough to run here or in thread)
        # But process_photos_batch expects a list, so let's find them first.
        # We'll do it in the executor to be safe.
        
        loop = asyncio.get_running_loop()
        
        try:
            photos = await loop.run_in_executor(None, find_jpeg_photos, self.directory)
            
            if not photos:
                self.status_label.text = "No photos found."
                self.is_running = False
                self.start_button.enabled = True
                self.folder_button.enabled = True
                return

            self.total_count = len(photos)
            self.progress_bar.max = self.total_count
            self.status_label.text = f"Found {self.total_count} photos. Loading model..."

            # Run the batch processing
            await loop.run_in_executor(
                None,
                process_photos_batch,
                photos,
                self.model,
                self.output_dir,
                self.status_callback_shim,
                self.stop_event,
                self.api_base,
                self.api_key
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

    def load_preview_image(self, path: Path) -> toga.Image:
        """Load an image for preview, resizing it to avoid UI overflow."""
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                # Resize to max 1920 width/height to prevent massive UI expansion
                # while maintaining good quality for preview
                img.thumbnail((1920, 1080))
                
                # Save to bytes
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'JPEG')
                return toga.Image(src=img_byte_arr.getvalue())
        except Exception as e:
            logger.warning(f"Failed to load preview {path}: {e}")
            return toga.Image(str(path))

    def status_callback_shim(self, processed, total, current_photo, last_analysis, comparison_photo=None):
        """Shim to call update_ui from the background thread."""
        self.loop.call_soon_threadsafe(self.update_ui, processed, total, current_photo, last_analysis, comparison_photo)

    def update_ui(self, processed, total, current_photo, last_analysis, comparison_photo=None):
        """Update UI elements on the main thread."""
        self.processed_count = processed
        self.progress_bar.value = processed
        
        # Calculate stats
        if self.initial_processed_count is None:
            self.initial_processed_count = processed
            
        delta_processed = processed - self.initial_processed_count
        
        avg_speed = 45.0 # Default
        if delta_processed > 0 and self.analysis_start_time:
            elapsed = time.time() - self.analysis_start_time
            avg_speed = elapsed / delta_processed
            
        remaining = total - processed
        eta_seconds = remaining * avg_speed
        
        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        else:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            
        status_text = f"Processing: {processed}/{total} | Avg: {avg_speed:.1f}s/photo | ETA: {eta_str}"
        self.status_label.text = status_text
        
        # Only update main view if NOT in review mode
        if not self.is_review_mode:
            if current_photo:
                if comparison_photo:
                    self.photo_label.text = f"Comparing: {comparison_photo.name} vs {current_photo.name}"
                    # Ensure both views are in container
                    if self.image_view_2 not in self.images_container.children:
                        self.images_container.add(self.image_view_2)
                    
                    try:
                        self.image_view.image = self.load_preview_image(comparison_photo)
                        self.image_view_2.image = self.load_preview_image(current_photo)
                    except Exception as e:
                        logger.warning(f"Failed to load comparison images: {e}")
                else:
                    self.photo_label.text = current_photo.name
                    # Ensure only main view is in container
                    if self.image_view_2 in self.images_container.children:
                        self.images_container.remove(self.image_view_2)

                    try:
                        self.image_view.image = self.load_preview_image(current_photo)
                    except Exception as e:
                        logger.warning(f"Failed to load image preview for {current_photo}: {e}")

        if last_analysis:
            self.add_recent_photo(last_analysis)

    def open_focus_check(self, widget):
        if hasattr(self, 'current_review_analysis') and self.current_review_analysis:
            window = FocusCheckWindow(self, self.current_review_analysis)
            window.show()

    def show_details(self, analysis: PhotoAnalysis):
        """Show details for a specific photo."""
        self.is_review_mode = True
        self.resume_button.enabled = True
        self.current_review_analysis = analysis
        self.focus_check_btn.enabled = True
        
        # Update main image
        try:
            abs_path = str(analysis.photo_path.resolve())
            
            # Overlay bounding box if available
            if analysis.primary_subject_bounding_box and len(analysis.primary_subject_bounding_box) == 4:
                try:
                    with Image.open(abs_path) as img:
                        img = ImageOps.exif_transpose(img)
                        # Handle EXIF rotation if needed (PIL usually handles it if we use ImageOps.exif_transpose, 
                        # but for now let's assume basic load. Toga might handle rotation on display, 
                        # but drawing on raw pixels might mismatch if we don't respect EXIF. 
                        # However, simple drawing is safer than re-saving with potential rotation loss.)
                        # Actually, if we save it back, we lose EXIF unless we copy it. 
                        # Let's just draw on what we have.
                        
                        draw = ImageDraw.Draw(img)
                        width, height = img.size
                        xmin, ymin, xmax, ymax = analysis.primary_subject_bounding_box
                        
                        # Convert 0-1000 scale to pixels
                        left = int((xmin / 1000) * width)
                        top = int((ymin / 1000) * height)
                        right = int((xmax / 1000) * width)
                        bottom = int((ymax / 1000) * height)
                        
                        # Draw red rectangle
                        outline_color = "#00FF00" # Lime
                        if analysis.blur_level == 1:
                            outline_color = "#FFA500" # Orange
                        elif analysis.blur_level == 2:
                            outline_color = "red"
                        draw.rectangle([left, top, right, bottom], outline=outline_color, width=5)
                        
                        # Save to temp file
                        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                        # Resize before saving for display
                        img.thumbnail((1920, 1080))
                        img.save(tf.name)
                        tf.close()
                        
                        # Use the temp file for display
                        self.image_view.image = toga.Image(tf.name)
                        
                        # We should probably clean up this temp file later, but for now let's rely on OS or next overwrite
                except Exception as e:
                    logger.warning(f"Failed to draw bounding box: {e}")
                    # Fallback to original
                    self.image_view.image = self.load_preview_image(analysis.photo_path)
            else:
                self.image_view.image = self.load_preview_image(analysis.photo_path)

            self.photo_label.text = analysis.photo_path.name
            
            # Ensure only main view is in container
            if self.image_view_2 in self.images_container.children:
                self.images_container.remove(self.image_view_2)
        except Exception as e:
            logger.error(f"Failed to load image for details: {e}")

        # Update details panel
        details_text = f"Rating: {analysis.rating} Stars\n\n"
        details_text += f"Rating reason: {analysis.rating_reason}\n\n"
        details_text += f"Subject: {analysis.primary_subject}\n\n"
        details_text += f"Keywords:\n{', '.join(analysis.keywords)}\n\n"
        details_text += f"Description:\n{analysis.description}\n\n"
        if analysis.raw_response:
             details_text += f"Raw Response:\n{analysis.raw_response}"
        if analysis.provider:
            details_text += f"\n\nProvider: {analysis.provider}\n"

        self.details_content.value = details_text
        
        # Show details panel if not visible
        if self.details_panel not in self.main_box.children:
            self.main_box.add(self.details_panel)

    def resume_live_view(self, widget):
        """Resume live view updates."""
        self.is_review_mode = False
        self.resume_button.enabled = False
        self.current_review_analysis = None
        self.focus_check_btn.enabled = False
        
        # Hide details panel
        if self.details_panel in self.main_box.children:
            self.main_box.remove(self.details_panel)
            
        self.photo_label.text = "Resuming live view..."

    def add_recent_photo(self, analysis: PhotoAnalysis):
        """Add a thumbnail to the recent photos strip."""
        # Create a box for the thumbnail
        thumb_box = toga.Box(style=Pack(direction=COLUMN, width=120, margin=5))
        
        try:
            # Create thumbnail image
            with Image.open(analysis.photo_path) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((200, 200))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'JPEG')
                toga_img = toga.Image(src=img_byte_arr.getvalue())
            
            # Toga buttons don't support large icons, so we use an ImageView for the thumbnail
            # and a separate button for the action.
            image_view = toga.ImageView(image=toga_img, style=Pack(height=80, width=100))
            
            if "BestInBurst" in analysis.keywords:
                # Wrap in red box for border effect
                border_box = toga.Box(style=Pack(background_color="red", margin=2))
                border_box.add(image_view)
                thumb_box.add(border_box)
            else:
                thumb_box.add(image_view)
            
            # Add View button below
            view_btn = toga.Button(
                "View", 
                on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
                style=Pack(width=100)
            )
            thumb_box.add(view_btn)
            
        except Exception as e:
            logger.warning(f"Failed to create thumbnail view: {e}")
            # Fallback to text button
            view_widget = toga.Button(
                "View", 
                on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
                style=Pack(height=80, width=100)
            )
            if "BestInBurst" in analysis.keywords:
                border_box = toga.Box(style=Pack(background_color="red", margin=2))
                border_box.add(view_widget)
                thumb_box.add(border_box)
            else:
                thumb_box.add(view_widget)
            
        rating_label = toga.Label(f"{analysis.rating} ★", style=Pack(text_align=CENTER))
        
        subject_text = analysis.primary_subject or ""
        if len(subject_text) > 15:
            subject_text = subject_text[:13] + ".."
        subject_label = toga.Label(subject_text, style=Pack(text_align=CENTER, font_size=10))
        
        thumb_box.add(rating_label)
        thumb_box.add(subject_label)
        
        # Add to start of list
        self.recent_box.insert(0, thumb_box)

def main(directory: Optional[Path] = None, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL, api_base: Optional[str] = None, api_key: str = "lm-studio"):
    return TprsStatusApp(directory, output_dir, model, api_base, api_key)
