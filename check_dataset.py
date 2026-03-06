import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder dataset yang mau dicek (mis. outputs_aug_main/standardized)")
    ap.add_argument("--size", type=int, default=224, help="Target ukuran (224 atau 256)")
    ap.add_argument("--mode", default="RGB", help="Target channel: RGB atau L (grayscale)")
    ap.add_argument("--out", default="spec_check.csv", help="Output CSV report")
    args = ap.parse_args()

    root = Path(args.root)
    target_size = (args.size, args.size)
    target_mode = args.mode.upper()

    rows = []
    total = 0
    bad_ext = 0
    corrupt = 0
    bad_size = 0
    bad_mode = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        total += 1
        ext = p.suffix.lower()
        ext_ok = ext in VALID_EXT
        if not ext_ok:
            bad_ext += 1

        try:
            with Image.open(p) as im:
                img_format = im.format  # JPEG / PNG / dll
                mode = im.mode          # RGB / L / RGBA / dll
                w, h = im.size

                # pixel stats (untuk sanity check)
                arr = np.array(im)
                pix_min = int(arr.min()) if arr.size else ""
                pix_max = int(arr.max()) if arr.size else ""

                size_ok = (w, h) == target_size
                mode_ok = (mode == target_mode)

                if not size_ok:
                    bad_size += 1
                if not mode_ok:
                    bad_mode += 1

                rows.append({
                    "path": str(p),
                    "ext": ext,
                    "format": img_format,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "size_ok": size_ok,
                    "mode_ok": mode_ok,
                    "ext_ok": ext_ok,
                    "pixel_min": pix_min,
                    "pixel_max": pix_max,
                    "status": "ok"
                })

        except Exception as e:
            corrupt += 1
            rows.append({
                "path": str(p),
                "ext": ext,
                "format": "",
                "width": "",
                "height": "",
                "mode": "",
                "size_ok": False,
                "mode_ok": False,
                "ext_ok": ext_ok,
                "pixel_min": "",
                "pixel_max": "",
                "status": f"corrupt: {type(e).__name__}"
            })

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)

    print("=== SUMMARY ===")
    print("Root:", root)
    print("Total files scanned:", total)
    print("Corrupt/unreadable:", corrupt)
    print("Bad extension:", bad_ext)
    print("Bad size:", bad_size, f"(expected {target_size})")
    print("Bad mode:", bad_mode, f"(expected {target_mode})")
    print("Report saved to:", out_path.resolve())

if __name__ == "__main__":
    main()