"""Chuẩn bị dữ liệu cho Lab 3 (Bài thực hành 4).

Tải 10 ảnh gốc từ repo mẫu OpenCV -> data/raw,
sinh ảnh biến đổi + bảng cặp (positive/negative) -> data/processed/pairs.csv.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import wavelet_hash as wh

SOURCES = [
    "baboon.jpg", "board.jpg", "building.jpg", "butterfly.jpg",
    "fruits.jpg", "home.jpg", "starry_night.jpg", "sudoku.png",
    "messi5.jpg", "aero1.jpg",
]
BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
SIZE = 256
TRANSFORMS = ["rot12", "scale08", "noise15", "bright30", "jpeg30"]


def download_raw(force=False):
    wh.RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in SOURCES if not (wh.RAW_DIR / f).exists() or force]
    import urllib.request

    for f in missing:
        urllib.request.urlretrieve(BASE_URL + f, wh.RAW_DIR / f)
        print(f"Tải: {f}")
    return [str(wh.RAW_DIR / f) for f in SOURCES]


def load_gray256(path):
    """Ảnh gốc đã resize SIZE x SIZE (BGR)."""
    img = wh.imread_unicode(path)
    assert img is not None, f"Không đọc được ảnh {path}"
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def apply_transform(img, tf):
    """Trả về ảnh biến đổi (một trong các loại TRANSFORMS)."""
    if tf == "rot12":  # xoay 12 độ, giữ kích thước
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if tf == "scale08":  # thu nhỏ 0.8 rồi phóng về kích thước ban đầu
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if tf == "noise15":  # nhiễu Gauss sigma 15
        noise = np.random.normal(0, 15.0, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    if tf == "bright30":  # tăng sáng 30% (≈ +76)
        return cv2.convertScaleAbs(img, alpha=1.0, beta=76)
    if tf == "jpeg30":  # nén JPEG mức 30
        return img
    raise ValueError(tf)


def build_dataset(seed=42, force=False):
    """Sinh 50 ảnh biến đổi + 50 cặp dương + 50 cặp âm -> pairs.csv."""
    if (wh.PROC_DIR / "pairs.csv").exists() and not force:
        return pd.read_csv(wh.PROC_DIR / "pairs.csv")

    raw_dir = wh.RAW_DIR
    out_dir = wh.PROC_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    src_imgs = {s: load_gray256(raw_dir / s) for s in SOURCES}

    # 1) luu tam? -> moi (source, transform) mot file
    rows = []  # (a, b, label, transform, source_a, source_b)
    for s in SOURCES:
        stem = Path(s).stem
        name = f"{stem}_orig.png"
        wh.imwrite_unicode(out_dir / name, src_imgs[s])
        for tf in TRANSFORMS:
            t_img = apply_transform(src_imgs[s], tf)
            out = out_dir / f"{stem}_{tf}.jpg" if tf == "jpeg30" else out_dir / f"{stem}_{tf}.png"
            if tf == "jpeg30":
                wh.imwrite_unicode(out, t_img, [cv2.IMWRITE_JPEG_QUALITY, 30])
            else:
                wh.imwrite_unicode(out, t_img)

            # positive: goc vs bien doi
            rows.append((name, out.name, 1, tf, s, s))
            # negative: bien doi cua nguon hien tai vs GOC nguon khac
            shifts = rng.permutation(len(SOURCES) - 1) + 1
            other = SOURCES[(SOURCES.index(s) + shifts[rng.integers(0, len(shifts))]) % len(SOURCES)]
            oname = f"{Path(other).stem}_orig.png"
            rows.append((out.name, oname, 0, tf, s, other))

    df = pd.DataFrame(rows, columns=["path_a", "path_b", "label", "transform", "source_a", "source_b"])
    df.to_csv(wh.PROC_DIR / "pairs.csv", index=False)
    print(f"Đã sinh {len(df)} cặp vào data/processed/pairs.csv")
    return df


if __name__ == "__main__":
    download_raw()
    df = build_dataset()
    print(df["label"].value_counts())
    print(df.head())
