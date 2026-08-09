Corrected annotation exports from Roboflow.

  batch01-train-451-corrected.yolov8.zip   451 train frames  (14 EyeCU matches)
  batch02-val-208-corrected.yolo26.zip     208 val frames    (4 frozen matches)

These are the raw human work product -- roughly 40 hours of manual correction.
Both are already imported into data/labels/ and validated, so nothing depends
on them day to day. They are kept as the off-tree copy: if data/labels is ever
corrupted or a remap goes wrong, these can be re-imported from scratch with

  python tools/import_roboflow.py --export <zip>

which re-applies the class remap and polygon conversion.

Note both live on the same disk as data/. That protects against a bad edit,
not against disk loss. Copy them somewhere else.

The external Roboflow datasets are NOT kept here -- they are re-downloadable
from Roboflow Universe, and their provenance is recorded in
data/external_provenance.json.
