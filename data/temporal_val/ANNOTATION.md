# Temporal validation benchmark — ball annotation

104 frames: two continuous windows per match, 5 FPS diagnostic sampling, 2s per match.

## What to label

The **ball only**. Players, goalkeepers and referees are not needed here — this benchmark exists to measure temporal ball recovery.

- One box on the ball when it is **visually identifiable**.
- If you cannot see it, leave the frame empty. An empty label file is a real signal: it is the ground truth for "no visible ball", which is what the selector must learn to report as `unknown` rather than inventing a position.
- **Do not** infer position from player gaze, from the previous frame, or from where the ball must be. A guessed ball makes the benchmark reward hallucination.
- If a second football is genuinely visible (spare ball, sideline), label it too and note the frame. The main val set never labels a second ball, so this benchmark is the only place that ambiguity can be resolved.

## Class ids

```
0 ball
```

Note this is a **benchmark-local** id. The main dataset uses `3 ball`; conversion happens at evaluation time, not here.

## Split

VALIDATION ONLY. Never train on these frames.
