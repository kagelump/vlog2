"""Stage 4: User Review Interface

This module provides a native macOS UI using Toga (BeeWare) for reviewing
AI-analyzed video clips and making final decisions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

import toga
from toga.style import Pack


class UserDecision(Enum):
    """User's decision for a clip."""

    APPROVE = "approve"  # Accept AI decision
    KEEP = "keep"  # Override: keep the clip
    REJECT = "reject"  # Override: reject the clip
    UNDECIDED = "undecided"  # Not yet reviewed


@dataclass
class ClipReviewItem:
    """A clip item for review in the UI."""

    source_path: Path
    thumbnail_path: Path | None
    camera_name: str
    duration_seconds: float
    ai_decision: str  # "keep", "reject", "review"
    confidence: str  # "high", "medium", "low"
    junk_reasons: list[str]
    user_decision: UserDecision = UserDecision.UNDECIDED
    suggested_in_point: float | None = None
    suggested_out_point: float | None = None


def create_review_app(
    clips: list[ClipReviewItem],
    on_complete: Callable[[list[ClipReviewItem]], None] | None = None,
) -> toga.App:
    """Create a Toga app for reviewing clips.

    Args:
        clips: List of clips to review.
        on_complete: Callback when user clicks "Generate Timeline".

    Returns:
        Toga App instance.
    """
    class ReviewApp(toga.App):
        """Toga application for clip review."""

        def __init__(self, clips: list[ClipReviewItem], on_complete_cb):
            super().__init__(
                formal_name="TVAS Review",
                app_id="com.tvas.review",
                app_name="TVAS Review",
            )
            self.clips = clips
            self.on_complete_cb = on_complete_cb
            self.current_filter = "all"

        def startup(self):
            """Build the main window."""
            # Main container
            main_box = toga.Box(style=Pack(direction="column", padding=10))

            # Header with stats
            stats = self._calculate_stats()
            self.stats_label = toga.Label(
                f"Clips: {stats['total']} | Keep: {stats['keep']} | "
                f"Reject: {stats['reject']} | Review: {stats['review']}",
                style=Pack(padding=(0, 0, 10, 0)),
            )
            main_box.add(self.stats_label)

            # Filter buttons
            filter_box = toga.Box(style=Pack(direction="row", padding=(0, 0, 10, 0)))
            filter_box.add(toga.Button("All", on_press=lambda w: self._set_filter("all")))
            filter_box.add(toga.Button("Keep", on_press=lambda w: self._set_filter("keep")))
            filter_box.add(toga.Button("Reject", on_press=lambda w: self._set_filter("reject")))
            filter_box.add(toga.Button("Review", on_press=lambda w: self._set_filter("review")))
            main_box.add(filter_box)

            # Scrollable clip list
            self.clip_container = toga.Box(style=Pack(direction="column", flex=1))
            scroll = toga.ScrollContainer(content=self.clip_container, style=Pack(flex=1))
            main_box.add(scroll)

            # Action buttons
            action_box = toga.Box(style=Pack(direction="row", padding=(10, 0, 0, 0)))
            action_box.add(
                toga.Button(
                    "Approve All AI Decisions",
                    on_press=self._approve_all,
                    style=Pack(padding=(0, 5)),
                )
            )
            action_box.add(
                toga.Button(
                    "Generate Timeline",
                    on_press=self._generate_timeline,
                    style=Pack(padding=(0, 5)),
                )
            )
            main_box.add(action_box)

            # Build clip items
            self._refresh_clips()

            # Create main window
            self.main_window = toga.MainWindow(title="TVAS - Review Clips")  # type: ignore[assignment]
            self.main_window.content = main_box  # type: ignore[attr-defined]
            self.main_window.show()  # type: ignore[attr-defined]

        def _calculate_stats(self) -> dict:
            """Calculate clip statistics."""
            return {
                "total": len(self.clips),
                "keep": sum(1 for c in self.clips if c.ai_decision == "keep"),
                "reject": sum(1 for c in self.clips if c.ai_decision == "reject"),
                "review": sum(1 for c in self.clips if c.ai_decision == "review"),
            }

        def _set_filter(self, filter_type: str):
            """Set the current filter."""
            self.current_filter = filter_type
            self._refresh_clips()

        def _refresh_clips(self):
            """Refresh the clip display based on current filter."""
            # Clear existing
            for child in list(self.clip_container.children):
                self.clip_container.remove(child)

            # Filter clips
            filtered = self.clips
            if self.current_filter != "all":
                filtered = [c for c in self.clips if c.ai_decision == self.current_filter]

            # Add clip items
            for clip in filtered:
                item_box = self._create_clip_item(clip)
                self.clip_container.add(item_box)

        def _create_clip_item(self, clip: ClipReviewItem) -> toga.Box:  # type: ignore[name-defined]
            """Create a UI element for a clip."""
            # Determine border color based on AI decision
            if clip.ai_decision == "reject":
                bg_color = "#ffcccc"  # Red tint
            elif clip.ai_decision == "review":
                bg_color = "#ffffcc"  # Yellow tint
            else:
                bg_color = "#ccffcc"  # Green tint

            item_box = toga.Box(
                style=Pack(
                    direction="row",
                    padding=5,
                    background_color=bg_color,
                )
            )

            # Clip info
            info_box = toga.Box(style=Pack(direction="column", flex=1))
            info_box.add(toga.Label(clip.source_path.name, style=Pack(font_weight="bold")))
            info_box.add(toga.Label(f"Camera: {clip.camera_name}"))
            info_box.add(toga.Label(f"Duration: {clip.duration_seconds:.1f}s"))
            info_box.add(toga.Label(f"AI: {clip.ai_decision} ({clip.confidence})"))

            if clip.junk_reasons:
                info_box.add(toga.Label(f"Issues: {', '.join(clip.junk_reasons)}"))

            if clip.suggested_in_point:
                info_box.add(toga.Label(f"Suggested trim: {clip.suggested_in_point}s start"))

            # User decision indicator
            decision_text = f"Your decision: {clip.user_decision.value}"
            info_box.add(toga.Label(decision_text))

            item_box.add(info_box)

            # Action buttons
            btn_box = toga.Box(style=Pack(direction="column", padding=(0, 0, 0, 10)))
            btn_box.add(
                toga.Button(
                    "Keep",
                    on_press=lambda w, c=clip: self._set_decision(c, UserDecision.KEEP),
                )
            )
            btn_box.add(
                toga.Button(
                    "Reject",
                    on_press=lambda w, c=clip: self._set_decision(c, UserDecision.REJECT),
                )
            )
            btn_box.add(
                toga.Button(
                    "Approve AI",
                    on_press=lambda w, c=clip: self._set_decision(c, UserDecision.APPROVE),
                )
            )
            item_box.add(btn_box)

            return item_box

        def _set_decision(self, clip: ClipReviewItem, decision: UserDecision):
            """Set user decision for a clip."""
            clip.user_decision = decision
            logger.info(f"Set decision for {clip.source_path.name}: {decision.value}")
            self._refresh_clips()
            self._update_stats()

        def _approve_all(self, widget):
            """Approve all AI decisions."""
            for clip in self.clips:
                if clip.user_decision == UserDecision.UNDECIDED:
                    clip.user_decision = UserDecision.APPROVE
            self._refresh_clips()
            self._update_stats()
            logger.info("Approved all AI decisions")

        def _update_stats(self):
            """Update the stats label."""
            stats = self._calculate_stats()
            self.stats_label.text = (
                f"Clips: {stats['total']} | Keep: {stats['keep']} | "
                f"Reject: {stats['reject']} | Review: {stats['review']}"
            )

        def _generate_timeline(self, widget):
            """Generate timeline and close."""
            logger.info("Generate Timeline clicked")
            if self.on_complete_cb:
                self.on_complete_cb(self.clips)
            self.main_window.close()  # type: ignore[attr-defined]

    return ReviewApp(clips, on_complete)


def run_review_ui(
    clips: list[ClipReviewItem],
    on_complete: Callable[[list[ClipReviewItem]], None] | None = None,
) -> list[ClipReviewItem]:
    """Run the review UI and return updated clips.

    This is a blocking call that returns when the user closes the UI.

    Args:
        clips: List of clips to review.
        on_complete: Optional callback when complete.

    Returns:
        Updated list of clips with user decisions.
    """
    app = create_review_app(clips, on_complete)
    app.main_loop()
    return clips


def create_review_items_from_analysis(
    analyses: list,  # List of ClipAnalysis from analysis module
) -> list[ClipReviewItem]:
    """Convert ClipAnalysis objects to ClipReviewItem for UI.

    Args:
        analyses: List of ClipAnalysis objects.

    Returns:
        List of ClipReviewItem objects.
    """
    from tvas.analysis import ClipAnalysis, ClipDecision

    items = []
    for analysis in analyses:
        if not isinstance(analysis, ClipAnalysis):
            continue

        # Map decision to string
        if analysis.decision == ClipDecision.KEEP:
            ai_decision = "keep"
        elif analysis.decision == ClipDecision.REJECT:
            ai_decision = "reject"
        else:
            ai_decision = "review"

        item = ClipReviewItem(
            source_path=analysis.source_path,
            thumbnail_path=None,  # TODO: Generate thumbnails
            camera_name=analysis.source_path.parent.name,
            duration_seconds=analysis.duration_seconds,
            ai_decision=ai_decision,
            confidence=analysis.confidence.value,
            junk_reasons=[r.value for r in analysis.junk_reasons],
            suggested_in_point=analysis.suggested_in_point,
            suggested_out_point=analysis.suggested_out_point,
        )
        items.append(item)

    return items


def show_notification(title: str, message: str) -> bool:
    """Show a system notification.

    Args:
        title: Notification title.
        message: Notification message.

    Returns:
        True if notification was shown.
    """
    # Toga notifications require a running app
    # For standalone notifications, we'd use a different approach
    logger.info(f"Notification: {title} - {message}")
    return True
