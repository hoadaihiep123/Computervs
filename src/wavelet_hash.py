"""Wavelet hash cho bài thực hành 4 - so sánh tương đồng ảnh bằng wavelet."""
from pathlib import Path

import cv2
import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"


def to_gray(image, size=128):
    """BGR/BGR-PIL ndarrayhay ảnh PIL -> grayscale, resize size x size."""
    if hasattr(image, "convert"):  # PIL Image
        image = np.array(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def imread_unicode(path):
    """cv2.imread đọc được đường dẫn chứa ký tự tiếng Việt (Chuyên Ngành)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, params=None):
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    assert ok, f"Không encode được {path}"
    buf.tofile(str(path))


def wavelet_hash(image, wavelet="db4", level=3, size=128):
    """Median-based hash trên subband LL (approximation).

    1. gray + resize 128x128
    2. DWT level-cấp -> lấy LL
    3. bit = 1 nếu hệ số > median(LL)
    Trả về mảng 1 chiều size/2**level * size/2**level bit.
    """
    g = to_gray(image, size)
    ll = pywt.wavedec2(g, wavelet, level=level)[0]
    return (ll > np.median(ll)).astype(np.uint8).ravel()


def wavelet_hash_quantize(image, wavelet="db4", level=3, size=128, step=2):
    """Dạng pseudocode trong slide: lượng tử hóa + bit % 2.

    Lưu ý: pywt.quantize không có trong PyWavelets hiện hành,
    lượng tử hóa chuẩn bằng np.round(coeff / step).
    """
    g = to_gray(image, size)
    ll = pywt.wavedec2(g, wavelet, level=level)[0]
    q = np.round(ll / step)
    return (q % 2).astype(np.uint8).ravel()


def hamming(hash1, hash2):
    """Khoảng cách Hamming chuẩn hoá về [0, 1]."""
    assert hash1.shape == hash2.shape
    return float(np.count_nonzero(hash1 != hash2) / hash1.size)


def pairwise_distances(hashes, pairs):
    """pairs: DataFrame/Liệt kê path_a/path_b -> array khoảng cách Hamming cùng thứ tự."""
    key_to_hash = {k: v for k, v in hashes.items()}
    return np.array([hamming(key_to_hash[a], key_to_hash[b]) for a, b in pairs.itertuples(index=False)])


def evaluate(distances, labels, threshold):
    """Tính confusion + metric độ chính xác chi tiết với ngưỡng cho trước.

    Dự đoán 2 ảnh tương đồng nếu distance <= threshold.
    """
    pred = (distances <= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    def safe_div(a, b):
        return a / b if b else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "recall (sensitivity)": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def pick_threshold(distances, labels):
    """Chọn ngưỡng trên tập train theo Youden's J (tối đa TPR - FPR).

    Score "giống nhau" = -distance (distance nhỏ => giống).
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, -np.asarray(distances, float))
    k = (tpr - fpr).argmax()
    return float(-thresholds[k])


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    print(f"Đã lưu: outputs/{name}")
