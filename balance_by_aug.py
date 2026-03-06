import os, csv, random
from pathlib import Path

import cv2
import numpy as np

VALID_EXT = (".jpg", ".jpeg", ".png")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXT],
                  key=lambda x: x.name.lower())

def simple_augment(img_bgr, rng: np.random.Generator):
    h, w = img_bgr.shape[:2]

    # rotate -20..20 derajat
    angle = float(rng.uniform(-20, 20))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # horizontal flip 50%
    do_flip = bool(rng.random() < 0.5)
    if do_flip:
        aug = cv2.flip(aug, 1)

    # brightness/contrast
    alpha = float(rng.uniform(0.8, 1.2))  # contrast
    beta  = float(rng.uniform(-20, 20))   # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # zoom/crop (random crop lalu resize balik)
    scale = float(rng.uniform(0.85, 1.0))
    new_h, new_w = int(h*scale), int(w*scale)
    y1 = int(rng.integers(0, h - new_h + 1))
    x1 = int(rng.integers(0, w - new_w + 1))
    aug = aug[y1:y1+new_h, x1:x1+new_w]
    aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

    meta = {"angle": angle, "flip": do_flip, "alpha": alpha, "beta": beta, "scale": scale}
    return aug, meta

def main():
    # === UBAH SESUAI FOLDER KAMU ===
    root = Path(r"outputs_aug_main\standardized")
    minor = root / "garpu"
    major = root / "sendok"   # kalau foldernya "Sendok", ganti jadi "Sendok"

    minor_files = list_images(minor)
    major_files = list_images(major)

    if not minor_files or not major_files:
        print("Folder minor/major kosong atau path salah.")
        print("minor:", minor)
        print("major:", major)
        return

    diff = len(major_files) - len(minor_files)
    if diff <= 0:
        print("Sudah balance atau minor sudah lebih banyak.")
        print("minor:", len(minor_files), "major:", len(major_files))
        return

    need = diff
    print(f"Minor: {len(minor_files)} | Major: {len(major_files)} | Tambah minor: {need}")

    rng = np.random.default_rng(1234)
    log_rows = []

    for i in range(need):
        base = random.choice(minor_files)
        img = cv2.imread(str(base))
        if img is None:
            continue

        aug, meta = simple_augment(img, rng)

        # pastikan nama file unik
        out_name = f"{base.stem}_bal{i+1:04d}.jpg"
        out_path = minor / out_name
        while out_path.exists():
            out_name = f"{base.stem}_bal{random.randint(1000,9999)}.jpg"
            out_path = minor / out_name

        cv2.imwrite(str(out_path), aug)

        log_rows.append({
            "base": base.name,
            "output": out_name,
            **meta
        })

    # simpan log augmentasi untuk laporan
    log_csv = root / "reports_balance.csv"
    ensure_dir(log_csv.parent)
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["base","output","angle","flip","alpha","beta","scale"])
        w.writeheader()
        w.writerows(log_rows)

    # hitung ulang
    minor_after = len(list_images(minor))
    major_after = len(list_images(major))
    print("SESUDAH:", "minor:", minor_after, "| major:", major_after)
    print("Log disimpan:", log_csv)

if __name__ == "__main__":
    main()