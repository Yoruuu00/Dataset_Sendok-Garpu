import os, csv, shutil, random
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
import imagehash

# ====== SETTING ======
DATA_ROOT = Path(r"Dataset_Augmented")
CLASS_DIRS = {
    "garpu": DATA_ROOT / "garpu",
    "sendok": DATA_ROOT / "Sendok",   # sesuaikan kalau folder kamu namanya beda
}
OUT_ROOT = Path("outputs_aug_main")

IMG_SIZE = (224, 224)   # boleh ganti (256,256)
FINAL_EXT = ".jpg"

# Threshold quality (boleh kamu sesuaikan & tulis di laporan)
BLUR_THR = 100.0        # variance of Laplacian
DARK_THR = 50           # mean brightness
BRIGHT_THR = 205

DUP_DIST_THR = 5        # pHash distance untuk near-duplicate
SAMPLE_PER_CLASS = 10   # untuk PPT

VALID_EXT = {".jpg", ".jpeg", ".png"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in VALID_EXT


def classify_type(filename: str):
    low = filename.lower()
    # deteksi augmented (umum): ada .rf. atau _aug atau aug_
    if ".rf." in low or "_aug" in low or "aug_" in low:
        return "augmented"
    return "original"


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


def standardize_and_save(src_path: Path, dst_path: Path):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    ensure_dir(dst_path.parent)
    img.save(dst_path, quality=95)


def write_csv(path: Path, fieldnames, rows):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    reports = OUT_ROOT / "reports"
    standardized = OUT_ROOT / "standardized"
    samples_dir = OUT_ROOT / "samples"

    ensure_dir(reports)
    ensure_dir(standardized)
    ensure_dir(samples_dir)

    # ===== (1) Collect files =====
    data = []  # (class, path)
    for cls, folder in CLASS_DIRS.items():
        if not folder.exists():
            print(f"[WARN] Folder kelas tidak ditemukan: {folder}")
            continue
        for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if is_image(p):
                data.append((cls, p))

    if not data:
        print("Tidak ada gambar ditemukan. Cek path folder kelas.")
        return

    # ===== (2) Count per class + original/augmented =====
    cnt_class = Counter([cls for cls, _ in data])
    cnt_type = Counter()
    cnt_type_per_class = defaultdict(Counter)

    for cls, p in data:
        t = classify_type(p.name)
        cnt_type[t] += 1
        cnt_type_per_class[cls][t] += 1

    write_csv(reports / "class_counts.csv", ["label", "count"],
              [{"label": k, "count": v} for k, v in cnt_class.items()])

    type_rows = []
    for cls in sorted(cnt_type_per_class.keys()):
        for t, v in cnt_type_per_class[cls].items():
            type_rows.append({"label": cls, "type": t, "count": v})
    write_csv(reports / "original_vs_augmented.csv", ["label", "type", "count"], type_rows)

    # ===== (3) Quality checks + corrupt =====
    quality_rows = []
    ok = []  # keep list
    for cls, p in data:
        img = cv2.imread(str(p))
        if img is None:
            quality_rows.append({"label": cls, "filename": p.name, "issue": "corrupt", "score": ""})
            continue

        issues = img_quality_checks(img)
        for issue, score in issues:
            quality_rows.append({"label": cls, "filename": p.name, "issue": issue, "score": score})

        ok.append((cls, p))

    write_csv(reports / "quality_issues.csv", ["label", "filename", "issue", "score"], quality_rows)

    # ===== (4) Duplicate / near-duplicate (pHash) =====
    hashes = []  # (cls, filename, hash)
    for cls, p in ok:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
            hashes.append((cls, p.name, h))
        except Exception:
            continue

    dup_rows = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            cls1, f1, h1 = hashes[i]
            cls2, f2, h2 = hashes[j]
            dist = h1 - h2
            if dist <= DUP_DIST_THR:
                dup_rows.append({
                    "file1": f1, "class1": cls1,
                    "file2": f2, "class2": cls2,
                    "distance": dist,
                    "note": "near-duplicate"
                })

    write_csv(reports / "duplicates.csv",
              ["file1", "class1", "file2", "class2", "distance", "note"],
              dup_rows)

    # ===== (5) Standardize =====
    for cls, p in ok:
        out_name = p.stem + FINAL_EXT
        dst = standardized / cls / out_name
        standardize_and_save(p, dst)

    # ===== (6) Sample images for PPT =====
    for cls in CLASS_DIRS.keys():
        cls_files = [p for c, p in ok if c == cls]
        if not cls_files:
            continue
        random.shuffle(cls_files)
        take = cls_files[:SAMPLE_PER_CLASS]
        out_cls = samples_dir / cls
        ensure_dir(out_cls)
        for p in take:
            shutil.copy2(p, out_cls / p.name)

    print("Selesai.")
    print(f"- Reports: {reports}")
    print(f"- Standardized: {standardized}")
    print(f"- Samples (untuk PPT): {samples_dir}")


if __name__ == "__main__":
    main()