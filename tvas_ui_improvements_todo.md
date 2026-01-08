# TVAS UI Improvements TODO

This document outlines UI improvements for the TVAS application, organized by impact and implementation difficulty.

---

## 🔴 High Impact / Quick Wins

### 1. ✅ Pipeline Buttons Reorganization
**Problem:** The left sidebar has individual phase buttons mixed with "Run All" buttons, making the hierarchy unclear. The disabled state of buttons isn't visually distinct enough.

**Solution:**
- Group individual phase buttons under a labeled "Individual Phases" section
- Add visual separators (dividers) between logical groups
- Add a "Run All (1-5)" button at the top for complete pipeline execution
- Make "Run Ingestion (1-3)" and "Run Post-Processing (4-5)" more prominent with better styling
- Add numbered step indicators with better visual hierarchy

### 2. ✅ Progress Section Enhancement
**Problem:** The progress bar, status, and ETA are functional but could be more informative.

**Solution:**
- Add a "Stop" button to cancel running operations
- Show current phase name more prominently above progress bar
- Add elapsed time display alongside ETA

### 3. ✅ Path Row Consistency
**Problem:** SD Card row has "Browse" + "Detect", Project has "Browse" + "Detect", Proxy only has "Browse", Outline only has "Browse". Inconsistent layout causes visual imbalance.

**Solution:**
- Add placeholder/disabled button to Proxy and Outline rows to maintain alignment
- Or resize the input fields to account for the missing buttons

---

## 🟡 Medium Impact

### 4. Recent Clips Thumbnails
**Problem:** The "View" buttons in Recent Clips are placeholders without actual thumbnails, reducing visual feedback.

**Solution:**
- Load thumbnails lazily in background after processing completes
- Cache thumbnails to disk to avoid re-extraction
- Show a loading spinner while thumbnail loads

### 5. Details Panel as Slide-out
**Problem:** Details panel appears/disappears suddenly and takes space from the main content area.

**Solution:**
- Make details panel a fixed-width sidebar that's always visible (but collapsed by default)
- Add a toggle button to show/hide
- Or make it a slide-in overlay from the right

### 6. Pipeline Status Indicators
**Problem:** Users can't easily see which phases have completed or are pending.

**Solution:**
- Add status icons (checkmark, spinner, pending) next to each phase button
- Color-code buttons based on state: gray (pending), blue (running), green (complete), red (error)

### 7. Keyboard Shortcuts
**Problem:** No keyboard shortcuts for common operations.

**Solution:**
- Cmd+1 through Cmd+5 for individual phases
- Cmd+R to run ingestion pipeline
- Cmd+P to run post-processing pipeline
- Cmd+S for settings

---

## 🟢 Lower Priority / Polish

### 8. Dark Mode Support
**Problem:** App appears to use system defaults, but explicit dark mode styling would improve appearance.

**Solution:**
- Detect system appearance and adjust colors accordingly
- Use darker background for main content area in dark mode

### 9. Recent Clips Sorting
**Problem:** Recent clips are sorted by processing order, not by filename or duration.

**Solution:**
- Add a sort dropdown (by name, duration, or processing order)
- Allow filtering by clip type or beat assignment

### 10. Drag & Drop Path Selection
**Problem:** Users must use file dialogs to select paths.

**Solution:**
- Allow drag-and-drop of folders onto path input fields
- Auto-detect folder type (SD card, project folder)

### 11. Settings Persistence
**Problem:** Settings only apply for the current session.

**Solution:**
- Save settings to a config file (~/.tvas/config.json)
- Load settings on startup
- Add "Reset to Defaults" button

### 12. Batch Queue View
**Problem:** During processing, there's no visibility into the queue of clips waiting.

**Solution:**
- Add a collapsible queue panel showing pending/completed/failed clips
- Allow clicking on queue items to view details

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Pipeline Buttons Reorganization | Low | High |
| 2 | Progress Section Enhancement (Stop button) | Low | High |
| 3 | Path Row Alignment | Low | Medium |
| 4 | Pipeline Status Indicators | Medium | High |
| 5 | Recent Clips Thumbnails | Medium | Medium |
| 6 | Settings Persistence | Medium | Medium |
| 7 | Keyboard Shortcuts | Low | Medium |
| 8 | Details Panel as Slide-out | Medium | Low |
| 9 | Dark Mode Support | Medium | Low |
| 10 | Drag & Drop | High | Low |

---

## Notes

- All changes should maintain Toga compatibility
- Test on macOS primarily as that's the deployment target
- Consider performance impact of any thumbnail-related changes
