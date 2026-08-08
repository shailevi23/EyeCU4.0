Review the successful YOLO26s @ 960 pilot before starting the full 80-epoch run.

Pilot status:
- Training completed successfully on Tesla T4.
- AutoBatch settled on batch=5 after probing batch=8 and hitting OOM; this is acceptable.
- After 10 epochs:
  - mAP50-95: ~0.45
  - player recall: ~0.91
  - ball recall: ~0.57
  - referee recall: ~0.50
  - goalkeeper recall: ~0.12
- Metrics were still improving at epoch 10, so continuing training is justified.

Before starting Experiment A:

1. Clean duplicate annotations
- Ultralytics reported duplicate labels in several train images and one val image.
- Identify and remove exact duplicate boxes from the source dataset.
- Re-run annotation validation afterward.
- Do not change legitimate overlapping-player boxes.

2. Check class balance
- Report instance counts for player / goalkeeper / referee / ball in:
  - train
  - val
  - test
- Pay special attention to goalkeeper training count.
- Do not alter val/test just to improve metrics.

3. Verify match-level leakage
- Confirm that no original match/source appears in more than one of train/val/test.
- The current notebook only checks image filename overlap; add/perform a source-match leakage check.

4. If all checks pass, give a GO/NO-GO for:
   train('A_yolo26s_960', 'yolo26s.pt', imgsz=960, epochs=80)

Do NOT:
- run Experiment B or C yet
- tune ByteTrack
- modify detector architecture
- use the test split for model selection

Return only:
- duplicates fixed
- class counts by split
- leakage result
- validation result
- GO / NO-GO for Experiment A