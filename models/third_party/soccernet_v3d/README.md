# SoccerNet-v3D ball detector (third-party weight)

EyeCU's production **ball** branch. Humans stay on the EyeCU YOLO detector; this
model supplies the ball class only.

```
official source   https://github.com/mguti97/SoccerNet-v3D
release           v1.0.0 ("Annotations and weights")
asset             yolo-sn-ball.pt
SHA256            e8c1a900300893c34bf36c964c5854ed93603470e04a4a8eba73f70e4eea148b
size              51,276,498 bytes
architecture      YOLO11l, 25,311,251 params, nc=1, names {0: 'ball'}
training imgsz    1280 (from the checkpoint's own train_args)
inference         imgsz 1280, conf >= 0.25
```

**The weight file is NOT tracked in git** — it is a 51 MB third-party artifact.
Obtain it separately from the official release above and place it here as
`yolo-sn-ball.pt`. `trackers/detector.py` verifies the SHA256 before the model
runs and refuses to start on a mismatch.

## Do not substitute the other release assets

The same release ships two fine-tuned siblings. Neither is a valid substitute:

- `yolo-sn-ball-opt.pt` — fine-tuned on SoccerNet-v3D optimized boxes. Measured
  in experiment S1D and found **materially worse** on EyeCU: temporal recall
  44/77 against this checkpoint's 59/77, hard contact 10/24 against 12/24,
  temporal IoU 27/77 against 57/77. It is architecturally identical and
  byte-different, so only the pinned hash distinguishes them.
- `yolo-issia-ball-opt.pt` — fine-tuned on ISSIA-v3D optimized boxes. Not
  evaluated for EyeCU.

## Licence

Recorded as found, with no legal conclusion drawn:

- the GitHub repository is reported as **GPL-2.0**
- the release page attaches no separate licence text to the weight assets
- the checkpoint carries an embedded field
  `license = 'AGPL-3.0 (https://ultralytics.com/license)'`, the stamp Ultralytics
  writes into every checkpoint it trains — distinct from the repository licence

Confirm the terms that apply to your use before redistributing this weight.
