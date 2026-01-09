# For TVAS

## Timestamp adjustment window
A lot of the time since this involves a lot of international travel, timestamp and timezones are all wonky.  There should be a window/flow after doing SD card copy -> Archival (or into proxy) that can make fixes and adjustments to timestamps so that they are all consistent.

## Music Sync / Rhythm:
The user mentioned "create beats". The current implementation implies "Story Beats" (narrative sections). It lacks musical beat alignment.

Add: A feature to analyze a music track (beat detection) and place markers in the generated Resolve timeline so clips snap to the rhythm.

## Iterative Outline Refinement:
The UI creates an outline once. A "Chat with your footage" interface would be better: "Show me all food clips but exclude the dark ones, and create a montage beat for them."

## Hybrid Processing:
Use a lightweight model (e.g., standard Computer Vision or a tiny VLM) to filter out clearly bad clips (blurry, pocket shots) before sending them to the expensive/slow high-quality VLM for description. (Note: The app already has some OpenCV blur checks, but could be smarter).

## Prompt Improvements

### video_describe.txt Enhancement
Add to the prompt:

```
- camera_movement: [static, pan, tilt, handheld, gimbal, running, walking]
- shot_type: [wide, medium, close-up, extreme_close, aerial, pov]
- b_roll_potential: 1-5 rating of how usable this is as generic B-roll
```

These help with beat alignment classification.

### beat_alignment.txt Enhancement
Add:

```
- suggested_duration: If this clip should be trimmed shorter than technical trim, how many seconds?
- alternative_beats: List 1-2 other beats this could work for
```

### video_trim.txt Enhancement
Currently only looks for bad starts/ends. Add:

```
- action_peaks: List timestamps where interesting action peaks occur
- dead_zones: List time ranges within the clip that are boring/redundant
```

This enables "highlight extraction" mode.