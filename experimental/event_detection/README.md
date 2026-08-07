# Experimental — Event Detection

`event_detector.py` detects match events (goals, shots, sprints) from tracking
data.

**Not production code.** The active pipeline is unchanged:

```text
run_pipeline.py → full_pipeline.py → trackers/
```

Nothing there imports this module. It is self-contained (numpy, cv2, stdlib
only) and kept for later evaluation. Integrating it is out of scope — see
TODO.md section 7.

It has never been run against real tracking data; it imports cleanly, which is
not the same as it working.
