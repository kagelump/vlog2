"""TPRS Status GUI

A non-interactive GUI for monitoring the Travel Photo Rating System progress.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

from tvas.tprs import PhotoAnalysis, process_photos_batch, find_jpeg_photos, DEFAULT_VLM_MODEL

# Configure logging to capture everything
logger = logging.getLogger("tvas")

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
    def __init__(self, directory: Path, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL):
        super().__init__("TPRS Status", "com.tvas.tprs_status")
        self.directory = directory
        self.output_dir = output_dir
        self.model = model
        self.processed_count = 0
        self.total_count = 0
        self.recent_photos = []  # List of (path, rating)
        self.is_running = False

    def startup(self):
        """Construct and show the Toga application."""
        
        # --- Header: Progress & Logs ---
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(padding=(0, 10), flex=1))
        self.status_label = toga.Label("Initializing...", style=Pack(padding=(5, 10)))
        self.log_label = toga.Label("Ready", style=Pack(padding=(0, 10), font_family="monospace", font_size=10))
        
        header_box = toga.Box(
            children=[self.status_label, self.progress_bar, self.log_label],
            style=Pack(direction=COLUMN, padding=10)
        )

        # --- Main: Current Photo ---
        
        # Use a flexible height and width to allow the image to scale properly
        self.image_view = toga.ImageView(style=Pack(flex=1))
        self.photo_label = toga.Label("No photo loaded", style=Pack(padding=5, text_align=CENTER))
        
        main_box = toga.Box(
            children=[self.image_view, self.photo_label],
            style=Pack(direction=COLUMN, flex=1, padding=10)
        )

        # --- Footer: Recent Photos ---
        self.recent_box = toga.Box(style=Pack(direction=ROW, padding=10, height=120))
        
        footer_container = toga.Box(
            children=[toga.Label("Recent Processed", style=Pack(padding=5)), self.recent_box],
            style=Pack(direction=COLUMN)
        )

        # --- Main Layout ---
        self.main_window = toga.MainWindow(title=self.formal_name, size=(1000, 800))
        self.main_window.content = toga.Box(
            children=[header_box, main_box, footer_container],
            style=Pack(direction=COLUMN)
        )
        
        # Setup Logging
        handler = GuiLogHandler(self)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.main_window.show()
        
        # Start processing automatically
        self.add_background_task(self.run_analysis)

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
                self.status_callback_shim
            )
            
            self.status_label.text = "Processing Complete!"
            self.progress_bar.value = self.total_count
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            self.status_label.text = f"Error: {e}"
        finally:
            self.is_running = False

    def status_callback_shim(self, processed, total, current_photo, last_analysis):
        """Shim to call update_ui from the background thread."""
        self.loop.call_soon_threadsafe(self.update_ui, processed, total, current_photo, last_analysis)

    def update_ui(self, processed, total, current_photo, last_analysis):
        """Update UI elements on the main thread."""
        self.processed_count = processed
        self.progress_bar.value = processed
        self.status_label.text = f"Processing: {processed}/{total}"
        
        if current_photo:
            self.photo_label.text = current_photo.name
            try:
                # Toga ImageView loads from path
                # Ensure path is absolute and string
                abs_path = str(current_photo.resolve())
                self.image_view.image = toga.Image(abs_path)
            except Exception as e:
                logger.warning(f"Failed to load image preview for {current_photo}: {e}")

        if last_analysis:
            self.add_recent_photo(last_analysis)

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
        
        thumb_box.add(view)
        thumb_box.add(rating_label)
        thumb_box.add(subject_label)
        
        # Add to start of list
        self.recent_box.insert(0, thumb_box)
        
        # Keep only last 5
        if len(self.recent_box.children) > 5:
            self.recent_box.remove(self.recent_box.children[-1])

def main(directory: Path, output_dir: Optional[Path] = None, model: str = DEFAULT_VLM_MODEL):
    return TprsStatusApp(directory, output_dir, model)
