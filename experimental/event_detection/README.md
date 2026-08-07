# Experimental — Event Detection

`event_detector.py` detects match events (goals, shots, sprints) from tracking
data. It is the one piece of unique functionality salvaged from the old
`HamzaIntegration/` (later renamed `Code/`) MVP; everything else there either
duplicated `trackers/`/`full_pipeline.py` or was on the TODO.md section 7
"do not work on" list.

**Not wired into the production pipeline.** The active path is unchanged:

```text
run_pipeline.py → full_pipeline.py → trackers/
```

Nothing here is imported by it. The module is self-contained (numpy, cv2,
stdlib only) and is kept for later evaluation, not current use. Integrating it
is explicitly out of scope for now.
