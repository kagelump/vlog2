"""TPRS Status GUI

A non-interactive GUI for monitoring the Travel Photo Rating System progress.
"""

import asyncio
import logging
import threading
import functools
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from tvas.tprs import PhotoAnalysis, process_photos_batch, find_jpeg_photos
from tvas import DEFAULT_VLM_MODEL

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
            self.app.log_label.text = msg


class TprsStatusApp(toga.App):
    def __init__(self, directory: Optional[Path] = None, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL):
        super().__init__("TPRS Status", "com.tvas.tprs_status")
        self.directory = directory
        self.output_dir = output_dir
        self.model = model
        self.processed_count = 0
        self.total_count = 0
        self.recent_photos = []  # List of (path, rating)
        self.is_running = False
        self.is_review_mode = False  # Track if user is viewing details
        self.stop_event = threading.Event()
        self.on_exit = self.exit_handler

    def exit_handler(self, app):
        """Handle app exit."""
        if self.is_running:
            logger.info("Stopping analysis...")
            self.stop_event.set()
            # Give it a moment? No, just return True and let it close.
            # The background thread will stop eventually.
        return True

    def startup(self):
        """Construct and show the Toga application."""
        
        # --- Control Panel: Folder Selection and Start Button ---
        self.folder_input = toga.TextInput(
            readonly=True,
            placeholder="Select a folder to scan...",
            style=Pack(flex=1, padding=(0, 5))
        )
        if self.directory:
            self.folder_input.value = str(self.directory)
        
        self.folder_button = toga.Button(
            "Browse...",
            on_press=self.select_folder,
            style=Pack(padding=(0, 5))
        )
        
        self.start_button = toga.Button(
            "Start Analysis",
            on_press=self.start_analysis,
            enabled=self.directory is not None,
            style=Pack(padding=(0, 5))
        )
        
        folder_row = toga.Box(
            children=[
                toga.Label("Folder:", style=Pack(padding=(5, 5), width=60)),
                self.folder_input,
                self.folder_button,
                self.start_button
            ],
            style=Pack(direction=ROW, padding=5)
        )
        
        # --- Header: Progress & Logs ---
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(padding=(0, 10), flex=1))
        self.status_label = toga.Label("Ready to start", style=Pack(padding=(5, 10)))
        self.log_label = toga.Label("Select a folder and click Start Analysis", style=Pack(padding=(0, 10), font_family="monospace", font_size=10, flex=1))
        
        self.resume_button = toga.Button(
            "Resume Live View",
            on_press=self.resume_live_view,
            enabled=False,
            style=Pack(padding=(0, 5))
        )

        log_row = toga.Box(
            children=[self.log_label, self.resume_button],
            style=Pack(direction=ROW, alignment=CENTER)
        )
        
        header_box = toga.Box(
            children=[self.status_label, self.progress_bar, log_row],
            style=Pack(direction=COLUMN, padding=10)
        )

        # --- Main: Current Photo & Details ---
        
        # Use a flexible height and width to allow the image to scale properly
        self.image_view = toga.ImageView(style=Pack(flex=1))
        self.image_view_2 = toga.ImageView(style=Pack(flex=1))
        
        self.images_container = toga.Box(
            children=[self.image_view],
            style=Pack(direction=ROW, flex=1)
        )
        
        self.photo_label = toga.Label("No photo loaded", style=Pack(padding=5, text_align=CENTER))
        
        self.image_area = toga.Box(
            children=[self.images_container, self.photo_label],
            style=Pack(direction=COLUMN, flex=1)
        )

        # Details Panel (Hidden by default or empty)
        self.details_label = toga.Label("Details", style=Pack(font_weight='bold', padding_bottom=5))
        self.details_content = toga.MultilineTextInput(readonly=True, style=Pack(flex=1))
        
        self.details_panel = toga.Box(
            children=[self.details_label, self.details_content],
            style=Pack(direction=COLUMN, width=300, padding=10)
        )
        # Initially hide details panel by removing it or setting width 0? 
        # Toga doesn't support hiding easily, so we'll just not add it to main_box initially?
        # Or we can just keep it there but empty.
        
        main_box = toga.Box(
            children=[self.image_area], # details_panel added dynamically
            style=Pack(direction=ROW, flex=1, padding=10)
        )
        self.main_box = main_box # Keep reference

        # --- Footer: Recent Photos ---
        self.recent_box = toga.Box(style=Pack(direction=ROW, padding=10))
        
        self.recent_scroll = toga.ScrollContainer(
            horizontal=True,
            vertical=False,
            style=Pack(height=150, flex=1)
        )
        self.recent_scroll.content = self.recent_box
        
        footer_container = toga.Box(
            children=[toga.Label("Recent Processed", style=Pack(padding=5)), self.recent_scroll],
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
                self.stop_event
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

    def status_callback_shim(self, processed, total, current_photo, last_analysis, comparison_photo=None):
        """Shim to call update_ui from the background thread."""
        self.loop.call_soon_threadsafe(self.update_ui, processed, total, current_photo, last_analysis, comparison_photo)

    def update_ui(self, processed, total, current_photo, last_analysis, comparison_photo=None):
        """Update UI elements on the main thread."""
        self.processed_count = processed
        self.progress_bar.value = processed
        self.status_label.text = f"Processing: {processed}/{total}"
        
        # Only update main view if NOT in review mode
        if not self.is_review_mode:
            if current_photo:
                if comparison_photo:
                    self.photo_label.text = f"Comparing: {comparison_photo.name} vs {current_photo.name}"
                    # Ensure both views are in container
                    if self.image_view_2 not in self.images_container.children:
                        self.images_container.add(self.image_view_2)
                    
                    try:
                        self.image_view.image = toga.Image(str(comparison_photo.resolve()))
                        self.image_view_2.image = toga.Image(str(current_photo.resolve()))
                    except Exception as e:
                        logger.warning(f"Failed to load comparison images: {e}")
                else:
                    self.photo_label.text = current_photo.name
                    # Ensure only main view is in container
                    if self.image_view_2 in self.images_container.children:
                        self.images_container.remove(self.image_view_2)

                    try:
                        # Toga ImageView loads from path
                        # Ensure path is absolute and string
                        abs_path = str(current_photo.resolve())
                        self.image_view.image = toga.Image(abs_path)
                    except Exception as e:
                        logger.warning(f"Failed to load image preview for {current_photo}: {e}")

        if last_analysis:
            self.add_recent_photo(last_analysis)

    def show_details(self, analysis: PhotoAnalysis):
        """Show details for a specific photo."""
        self.is_review_mode = True
        self.resume_button.enabled = True
        
        # Update main image
        try:
            abs_path = str(analysis.photo_path.resolve())
            
            # Overlay bounding box if available
            if analysis.primary_subject_bounding_box and len(analysis.primary_subject_bounding_box) == 4:
                try:
                    with Image.open(abs_path) as img:
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
                        draw.rectangle([left, top, right, bottom], outline="red", width=5)
                        
                        # Save to temp file
                        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                        img.save(tf.name)
                        tf.close()
                        
                        # Use the temp file for display
                        self.image_view.image = toga.Image(tf.name)
                        
                        # We should probably clean up this temp file later, but for now let's rely on OS or next overwrite
                except Exception as e:
                    logger.warning(f"Failed to draw bounding box: {e}")
                    # Fallback to original
                    self.image_view.image = toga.Image(abs_path)
            else:
                self.image_view.image = toga.Image(abs_path)

            self.photo_label.text = analysis.photo_path.name
            
            # Ensure only main view is in container
            if self.image_view_2 in self.images_container.children:
                self.images_container.remove(self.image_view_2)
        except Exception as e:
            logger.error(f"Failed to load image for details: {e}")

        # Update details panel
        details_text = f"Rating: {analysis.rating} Stars\n\n"
        details_text += f"Subject: {analysis.primary_subject}\n\n"
        details_text += f"Keywords:\n{', '.join(analysis.keywords)}\n\n"
        details_text += f"Description:\n{analysis.description}\n\n"
        if analysis.raw_response:
             details_text += f"Raw Response:\n{analysis.raw_response[:500]}..."

        self.details_content.value = details_text
        
        # Show details panel if not visible
        if self.details_panel not in self.main_box.children:
            self.main_box.add(self.details_panel)

    def resume_live_view(self, widget):
        """Resume live view updates."""
        self.is_review_mode = False
        self.resume_button.enabled = False
        
        # Hide details panel
        if self.details_panel in self.main_box.children:
            self.main_box.remove(self.details_panel)
            
        self.photo_label.text = "Resuming live view..."

    def add_recent_photo(self, analysis: PhotoAnalysis):
        """Add a thumbnail to the recent photos strip."""
        # Create a box for the thumbnail
        thumb_box = toga.Box(style=Pack(direction=COLUMN, width=100, padding=5))
        
        try:
            abs_path = str(analysis.photo_path.resolve())
            img = toga.Image(abs_path)
            view = toga.ImageView(image=img, style=Pack(height=80, width=100))
        except:
            view = toga.ImageView(style=Pack(height=80, width=100))
            
        rating_label = toga.Label(f"{analysis.rating} ★", style=Pack(text_align=CENTER))
        
        subject_text = analysis.primary_subject or ""
        if len(subject_text) > 15:
            subject_text = subject_text[:13] + ".."
        subject_label = toga.Label(subject_text, style=Pack(text_align=CENTER, font_size=10))
        
        # View button
        view_btn = toga.Button(
            "View", 
            on_press=functools.partial(lambda a, w: self.show_details(a), analysis),
            style=Pack(font_size=10, padding_top=2)
        )

        if "BestInBurst" in analysis.keywords:
            # Wrap in red box for border effect
            border_box = toga.Box(style=Pack(background_color="red", padding=2))
            border_box.add(view)
            thumb_box.add(border_box)
        else:
            thumb_box.add(view)

        thumb_box.add(rating_label)
        thumb_box.add(subject_label)
        thumb_box.add(view_btn)
        
        # Add to start of list
        self.recent_box.insert(0, thumb_box)

def main(directory: Optional[Path] = None, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL):
    return TprsStatusApp(directory, output_dir, model)
