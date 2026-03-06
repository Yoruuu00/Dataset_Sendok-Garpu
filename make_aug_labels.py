import os, csv

LABELS_ASLI = "labels/labels.csv"
AUG_DIR = r"Dataset_Augmented\Dataset"   # sesuai struktur repo kamu
OUT_CSV = "labels/labels_augmented.csv"

def load_asli_labels(path):
    m = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m[row["filename"]] = row["label"].strip().lower()
    return m

def base_from_aug_filename(aug_name: str):
    # contoh: IMG_..._jpg.rf.xxx.jpg -> IMG_....jpg
    if "_jpg.rf." in aug_name:
        return aug_name.split("_jpg.rf.")[0] + ".jpg"
    if ".jpg.rf." in aug_name:
        return aug_name.split(".jpg.rf.")[0] + ".jpg"
    if ".jpeg.rf." in aug_name:
        return aug_name.split(".jpeg.rf.")[0] + ".jpeg"
    if ".png.rf." in aug_name:
        return aug_name.split(".png.rf.")[0] + ".png"
    return None

def main():
    asli = load_asli_labels(LABELS_ASLI)
    rows = []

    for fn in os.listdir(AUG_DIR):
        low = fn.lower()
        if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
            continue

        base = base_from_aug_filename(fn)
        if base and base in asli:
            rows.append({"filename": fn, "label": asli[base], "source": base})
        else:
            rows.append({"filename": fn, "label": "unknown", "source": base or ""})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {OUT_CSV} ({len(rows)} rows)")

if __name__ == "__main__":
    main()