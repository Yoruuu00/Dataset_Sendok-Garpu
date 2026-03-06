import csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
import imagehash

# === Dataset FINAL yang diaudit ===
ROOT = Path(r"outputs_aug_main\standardized")
CLASS_DIRS = {
    "garpu": ROOT / "garpu",
    "sendok": ROOT / "sendok",
}

# === Output report baru (biar tidak menimpa yang lama) ===
OUT_DIR = Path(r"outputs_aug_main\reports_reaudit")

# === Threshold audit (sama seperti sebelumnya) ===
BLUR_THR = 100.0
DARK_THR = 50
BRIGHT_THR = 205
DUP_DIST_THR = 5

VALID_EXT = {".jpg", ".jpeg", ".png"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in VALID_EXT

def img_quality_checks(img_bgr):
    issues = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THR:
        issues.append(("blur_heavy", float(blur_score)))

    mean_bright = float(np.mean(gray))
    if mean_bright < DARK_THR:
        issues.append(("too_dark", mean_bright))
    if mean_bright > BRIGHT_THR:
        issues.append(("too_bright", mean_bright))

    return issues

def write_csv(path: Path, fieldnames, rows):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ensure_dir(OUT_DIR)

    # (A) Collect files
    data = []  # (label, path)
    for label, folder in CLASS_DIRS.items():
        if not folder.exists():
            print(f"[WARN] Folder tidak ditemukan: {folder}")
            continue
        for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if is_image(p):
                data.append((label, p))

    if not data:
        print("Tidak ada gambar untuk diaudit. Cek path folder.")
        return

    # (B) class counts
    cnt = Counter([label for label, _ in data])
    write_csv(OUT_DIR / "class_counts_reaudit.csv",
              ["label", "count"],
              [{"label": k, "count": v} for k, v in cnt.items()])

    # (C) quality audit + corrupt
    quality_rows = []
    ok = []  # (label, path) untuk lanjut duplicate
    for label, p in data:
        img = cv2.imread(str(p))
        if img is None:
            quality_rows.append({"label": label, "filename": p.name, "issue": "corrupt", "score": ""})
            continue

        issues = img_quality_checks(img)
        for issue, score in issues:
            quality_rows.append({"label": label, "filename": p.name, "issue": issue, "score": score})

        ok.append((label, p))

    write_csv(OUT_DIR / "quality_issues_reaudit.csv",
              ["label", "filename", "issue", "score"],
              quality_rows)

    # (D) duplicate audit (pHash)
    hashes = []  # (label, filename, hash)
    for label, p in ok:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
            hashes.append((label, p.name, h))
        except Exception:
            continue

    dup_rows = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            lab1, f1, h1 = hashes[i]
            lab2, f2, h2 = hashes[j]
            dist = h1 - h2
            if dist <= DUP_DIST_THR:
                dup_rows.append({
                    "file1": f1, "class1": lab1,
                    "file2": f2, "class2": lab2,
                    "distance": dist,
                    "note": "near-duplicate"
                })

    write_csv(OUT_DIR / "duplicates_reaudit.csv",
              ["file1", "class1", "file2", "class2", "distance", "note"],
              dup_rows)

    print("Audit ulang selesai.")
    print("Output:", OUT_DIR)

if __name__ == "__main__":
    main()