"""Verify what a model topology actually inherits from a pretrained donor.

Built for the B2/P2 experiment, where the detection topology changes and most
of the head necessarily re-initialises. The question that matters is not "did
loading succeed" -- Ultralytics loads whatever intersects and prints a count --
but *which* modules inherited and which did not.

Two figures are always reported together. Tensor percentage understates
transfer badly, because the Detect head contributes many small tensors while
the backbone holds a few very large convolutions; parameter percentage is the
meaningful one. Reporting either alone is misleading.

The hard gate is the backbone. Layers 0-10 are topology-independent between
yolo26 and yolo26-p2, so anything below 100% there means the donor, the scale
or the yaml is wrong -- not that transfer is merely suboptimal. A 0% Detect
head is expected and never fails the gate.

Matching mirrors Ultralytics' own path (`intersect_dicts`): same key, same
shape.

Usage:
    python tools/verify_weight_transfer.py --model-yaml models/yolo26s-p2-widthmatched.yaml \\
        --donor yolo26s.pt --nc 4 --json report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BACKBONE_LAST = 10  # layers 0-10 are the shared feature extractor


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_index(key: str) -> int:
    parts = key.split(".")
    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-yaml", required=True, type=Path)
    ap.add_argument("--donor", required=True, type=Path)
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--expect-backbone-pct", type=float, default=100.0)
    ap.add_argument("--expect-total-pct", type=float, default=None,
                    help="frozen total parameter transfer to reproduce, e.g. 58.19")
    ap.add_argument("--total-tolerance", type=float, default=0.05,
                    help="allowed absolute deviation in percentage points")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import yaml as pyyaml
    from ultralytics import YOLO

    # Build the receiver at the experiment's class count.
    spec = pyyaml.safe_load(args.model_yaml.read_text(encoding="utf-8"))
    spec["nc"] = args.nc
    tmp = args.model_yaml.parent / f"_nc{args.nc}_{args.model_yaml.name}"
    tmp.write_text(pyyaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    try:
        model = YOLO(str(tmp), task="detect").model.float()
    finally:
        tmp.unlink(missing_ok=True)

    ckpt = torch.load(args.donor, map_location="cpu", weights_only=False)
    donor = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    dsd = donor.float().state_dict()
    rsd = model.state_dict()

    detect_idx = len(model.model) - 1
    groups = {
        f"backbone (0-{BACKBONE_LAST})": [0, 0, 0, 0],
        "neck + new paths": [0, 0, 0, 0],
        "Detect head": [0, 0, 0, 0],
    }
    shape_mismatch, absent = [], []

    for k, v in rsd.items():
        i = layer_index(k)
        g = (f"backbone (0-{BACKBONE_LAST})" if 0 <= i <= BACKBONE_LAST
             else "Detect head" if i == detect_idx else "neck + new paths")
        if k not in dsd:
            absent.append(k)
            hit = False
        elif dsd[k].shape != v.shape:
            shape_mismatch.append(
                {"key": k, "donor": list(dsd[k].shape), "receiver": list(v.shape)})
            hit = False
        else:
            hit = True
        groups[g][0] += hit
        groups[g][1] += v.numel() if hit else 0
        groups[g][2] += 1
        groups[g][3] += v.numel()

    ok_t = sum(g[0] for g in groups.values())
    ok_p = sum(g[1] for g in groups.values())
    tot_t = sum(g[2] for g in groups.values())
    tot_p = sum(g[3] for g in groups.values())

    def pct(a, b):
        return round(100.0 * a / b, 4) if b else 0.0

    report = {
        "model_yaml": str(args.model_yaml),
        "nc": args.nc,
        "donor": str(args.donor),
        "donor_sha256": sha256_file(args.donor),
        "detect_layer_index": detect_idx,
        "total": {
            "tensors": f"{ok_t}/{tot_t}", "tensors_pct": pct(ok_t, tot_t),
            "parameters": f"{ok_p}/{tot_p}", "parameters_pct": pct(ok_p, tot_p),
        },
        "by_module": {
            g: {"tensors": f"{v[0]}/{v[2]}", "tensors_pct": pct(v[0], v[2]),
                "parameters": f"{v[1]}/{v[3]}", "parameters_pct": pct(v[1], v[3])}
            for g, v in groups.items()
        },
        "keys_absent_from_donor": len(absent),
        "shape_mismatches": len(shape_mismatch),
        "shape_mismatch_detail": shape_mismatch[:40],
    }

    print(f"model      : {args.model_yaml.name}  (nc={args.nc})")
    print(f"donor      : {args.donor.name}")
    print(f"donor sha  : {report['donor_sha256']}")
    print()
    print(f"  tensors    {ok_t}/{tot_t} = {pct(ok_t, tot_t):.2f}%")
    print(f"  parameters {ok_p:,}/{tot_p:,} = {pct(ok_p, tot_p):.2f}%   <- the meaningful figure")
    print()
    for g, v in groups.items():
        print(f"    {g:<22} {v[0]:>4}/{v[2]:<4} tensors  "
              f"{v[1]:>10,}/{v[3]:<10,} params  {pct(v[1], v[3]):>6.2f}%")
    print()
    print(f"  keys absent from donor : {len(absent)}")
    print(f"  shape mismatches       : {len(shape_mismatch)}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\n  report -> {args.json}")

    failures = []
    bb = groups[f"backbone (0-{BACKBONE_LAST})"]
    bb_pct = pct(bb[1], bb[3])
    if bb_pct < args.expect_backbone_pct:
        failures.append(
            f"backbone transfer {bb_pct:.2f}% < required {args.expect_backbone_pct:.2f}% "
            "-- donor, scale or yaml mismatch")
    if args.expect_total_pct is not None:
        got = pct(ok_p, tot_p)
        if abs(got - args.expect_total_pct) > args.total_tolerance:
            failures.append(
                f"total parameter transfer {got:.2f}% != frozen {args.expect_total_pct:.2f}% "
                f"(tolerance {args.total_tolerance})")

    if failures:
        print("\nTRANSFER GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 2
    print("\nTRANSFER GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
