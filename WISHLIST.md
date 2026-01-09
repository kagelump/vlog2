# For TVAS

## "Best Moment" Detection:
Currently, the app detects start/end junk. Improve this by asking the AI to return the timestamp of the "most aesthetic 3-5 second window" in the clip. This allows for an "Insta-cut" mode where it aggressively trims to the best moments.

## Music Sync / Rhythm:
The user mentioned "create beats". The current implementation implies "Story Beats" (narrative sections). It lacks musical beat alignment.

Add: A feature to analyze a music track (beat detection) and place markers in the generated Resolve timeline so clips snap to the rhythm.

## Iterative Outline Refinement:
The UI creates an outline once. A "Chat with your footage" interface would be better: "Show me all food clips but exclude the dark ones, and create a montage beat for them."

## Hybrid Processing:
Use a lightweight model (e.g., standard Computer Vision or a tiny VLM) to filter out clearly bad clips (blurry, pocket shots) before sending them to the expensive/slow high-quality VLM for description. (Note: The app already has some OpenCV blur checks, but could be smarter).

