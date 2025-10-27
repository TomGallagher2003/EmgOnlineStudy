---
title: Swap Movement Sets & Image Library
summary: Quick guide to change which movements appear and which images the app shows.
---

# Swap Movement Sets & Image Library

This page shows how to replace the default movement images, change which movements are in Set A / Set B, and keep labels consistent.

## TL;DR
- Edit `images.py` to point `MOVEMENT_IMAGES_A`, `MOVEMENT_IMAGES_B`, and `REST` to your own image files.
- Keep a consistent naming convention (e.g., `M1`…`M29`) so labels stay predictable.
- If you add or remove movements, update any hard‑coded counts in your timers/UI.

## Where to change things
- **File:** `images.py`
- **Keys to look for:** `MOVEMENT_IMAGES_A`, `MOVEMENT_IMAGES_B`, `REST`

```python
class Images:

    MOVEMENT_IMAGES_A = [
        "movement_library/EA/Index_flexion_M1.png",
        "movement_library/EA/Index_Extension_M2.png",
        "movement_library/EA/Middle_Flexion_M3.png",
      ...
        ]
    MOVEMENT_IMAGES_B = [
        "movement_library/EB/Thrumb_up_M13.png",
        "movement_library/EB/Extension_of_index_and_middle_M14.PNG.png",
        "movement_library/EB/Flexion_of_little_and_ring_M15.PNG.png",
    ...     
    ]

    REST = "movement_library/Rest_M0.png"
```

### Adding or replacing images
1. Drop your new images into your assets folder (e.g., `assets/movements/`).
2. Update the paths in `MOVEMENT_IMAGES_A/B` to match.
3. (Optional) Keep file names aligned to `<short_name>_M<number>.gif` to simplify mapping.

### Changing which movements appear
- Remove or reorder entries in `MOVEMENT_IMAGES_A/B` to control which movements show and in what order.
- If you change set lengths, confirm any code that assumes a fixed count (e.g., `len(MOVEMENT_IMAGES_A)` in a loop).

## Label sanity (important)
Many pipelines infer labels from **index position** in A/B:
- If **Set A index i** maps to label `i+1`, reordering changes labels.
- If you need stable labels across releases, either:
  - Keep order stable and only swap the underlying files, **or**
  - Maintain a separate `{filename → label}` mapping and use that to generate labels (you will need to make some changes in timer.py).

### Example: explicit mapping (advanced)
```python
# images.py
MOVEMENT_LABELS_A = {
    "assets/movements/M1_open_hand.gif": 1,
    "assets/movements/M2_close_fist.gif": 2,
    # ...
}
```
Then, when building your UI or saving labels, look up the label from the file path.

## Rest image / “get ready” screens
- `REST` typically points to a neutral image/GIF used between movements.
- You can swap this for a “Get Ready” or countdown graphic if preferred.

## Troubleshooting
- **Image doesn’t show?** Check the console for a file‑not‑found path.
- **Labels offset or wrong?** Revisit ordering and any `index_offset` logic in the timers.