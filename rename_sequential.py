import os
import csv
import argparse
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png"}

def list_images(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            files.append(p)
    # urutkan stabil (nama file)
    files.sort(key=lambda x: x.name.lower())
    return files

def two_phase_rename(files, new_names):
    # Phase 1: rename ke nama sementara supaya tidak bentrok
    temp_paths = []
    for i, src in enumerate(files):
        tmp = src.with_name(f"__tmp__rename__{i:06d}{src.suffix.lower()}")
        if tmp.exists():
            raise FileExistsError(f"Temp file already exists: {tmp}")
        src.rename(tmp)
        temp_paths.append(tmp)

    # Phase 2: rename ke nama final
    final_paths = []
    for tmp, final in zip(temp_paths, new_names):
        if final.exists():
            raise FileExistsError(f"Target already exists: {final}")
        tmp.rename(final)
        final_paths.append(final)

    return final_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder yang berisi gambar")
    ap.add_argument("--prefix", default="", help="Prefix nama file (mis: sendok atau garpu)")
    ap.add_argument("--start", type=int, default=1, help="Mulai penomoran (default 1)")
    ap.add_argument("--digits", type=int, default=4, help="Jumlah digit zero padding (default 4)")
    ap.add_argument("--dry-run", action="store_true", help="Preview saja, tidak rename")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    files = list_images(folder)
    if not files:
        print("Tidak ada file gambar yang cocok (.jpg/.jpeg/.png).")
        return

    prefix = args.prefix.strip()
    mapping = []

    new_paths = []
    for idx, src in enumerate(files, start=args.start):
        num = str(idx).zfill(args.digits)
        # format nama baru
        if prefix:
            new_name = f"{prefix}_{num}{src.suffix.lower()}"
        else:
            new_name = f"{num}{src.suffix.lower()}"  # kalau memang mau 0001.jpg dst
        new_paths.append(src.with_name(new_name))
        mapping.append((src.name, new_name))

    # cek duplikat nama target
    if len(set(p.name for p in new_paths)) != len(new_paths):
        raise RuntimeError("Ada nama target yang duplikat. Cek prefix/digits.")

    # print preview
    print(f"Folder: {folder}")
    print(f"Total gambar: {len(files)}")
    print("Contoh mapping (maks 10):")
    for old, new in mapping[:10]:
        print(f"  {old} -> {new}")

    if args.dry_run:
        print("\nDRY RUN aktif: tidak ada file yang di-rename.")
        return

    # pastikan tidak ada target yang sudah ada (biar aman)
    for p in new_paths:
        if p.exists():
            raise FileExistsError(f"Target sudah ada: {p.name} (hapus/rename dulu atau ubah prefix)")

    # lakukan rename aman (2-phase)
    two_phase_rename(files, new_paths)

    # simpan mapping csv
    map_csv = folder / "rename_mapping.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name"])
        w.writerows(mapping)

    print(f"\nSelesai rename. Mapping disimpan di: {map_csv}")

if __name__ == "__main__":
    main()