# Mô tả Lab 3 (Bài thực hành 4 — Chương 3: Phát hiện đặc trưng và đối sánh ảnh)

Nguồn: `Chapter3_Feature Detection and Image Matching_Part1.pdf` (slide 16–18) và `_Part2.pdf` (slide 8–9).

## Đề bài — Bài thực hành 4: So sánh sự tương đồng của các hình ảnh sử dụng Wavelet, Python

1. Chuẩn bị dữ liệu: tập ảnh gồm các cặp tương tự (cùng đối tượng, góc/mức nhiễu khác nhau) và cặp không tương tự.
2. Trích xuất wavelet: dùng biến đổi wavelet chuyển từng ảnh thành ma trận wavelet.
3. Tạo mã băm cho mỗi ảnh dựa trên các hệ số wavelet đã lượng tử hóa.
4. So sánh hash: tính khoảng cách Hamming giữa các mã băm để đánh giá độ tương đồng.
5. Đánh giá: accuracy, recall (sensitivity), specificity (độ đặc hiệu), precision, đường cong ROC.

Bài tập nâng cao:
6. Khảo sát các phương pháp băm wavelet khác nhau và so sánh hiệu suất.
7. Xây dựng ứng dụng tìm kiếm hình ảnh dựa trên hàm băm wavelet.

## Bài tập bổ sung (Part 2): Nhận diện khuôn mặt thời gian thực với FaceNet & MTCNN trên Webcam

Webcam (OpenCV) → MTCNN phát hiện khuôn mặt → FaceNet trích embedding → so cosine similarity,
ngưỡng của đề bài: similarity > 0.7 hiển thị "Matched", < 0.7 hiển thị "Unknown".

Ảnh chụp khi đăng ký được lưu vào `data/raw/faces/<tên>_<số thứ tự>.jpg`;
CSDL embedding lưu ở `data/raw/faces/faces_db.pt` (phiên sau nạp lại, không cần chụp lại).

## Cấu trúc thư mục

```text
Lab3/
├── data/
│   ├── raw/              # 10 ảnh gốc tải từ bộ mẫu OpenCV (+ lena.jpg dùng cho FaceNet)
│   └── processed/
│       ├── images/       # ảnh gốc 256×256 + 50 ảnh biến đổi (rot12, scale08, noise15, bright30, jpeg30)
│       └── pairs.csv     # 100 cặp: 50 dương (label=1) / 50 âm (label=0)
├── outputs/              # hình (fig_01..fig_07) + variant_results.csv
├── src/
│   ├── Lab3_code.ipynb     # notebook chính: 5 task + 2 task nâng cao (chạy từ trên xuống)
│   ├── Facenet_code.ipynb  # notebook FaceNet + MTCNN realtime (chạy trên máy có webcam)
│   ├── wavelet_hash.py     # hàm hash / hamming / evaluate / pick_threshold
│   └── prepare_data.py     # tải ảnh, sinh ảnh biến đổi và bộ cặp (seed cố định)
├── Task/
│   ├── Lab3_descript.md    # mô tả đề bài (file này)
│   └── report.docx         # (nếu nhóm cần nộp báo cáo)
├── README.md
└── requirements.txt
```

## Cách chạy

1. `pip install -r requirements.txt`
2. Mở `src/Lab3_code.ipynb` → Run All.
3. FaceNet: cần webcam, mở `src/Facenet_code.ipynb` chạy từng ô theo thứ tự (đăng ký khuôn mặt trước, rồi demo realtime).

## Ghi chú kỹ thuật so với pseudocode trong slide

- `pywt.wavedec2` trả về **tuple lồng nhau** (LL là mảng, detail là các tuple) — không flatten trực tiếp được; bản này chỉ dùng subband LL.
- Ảnh phải **resize về kích thước cố định (128×128)** trước khi hash, nếu không LL hai ảnh khác kích thước → không so Hamming được.
- `pywt.quantize` không có trong PyWavelets hiện hành — bản so sánh dùng `np.round(coeff / step)`.
- Metric: slide ghi "Độ đặc hiệu: precision" là nhầm — **độ đặc hiệu = specificity = TN/(TN+FP)**, còn precision = TP/(TP+FP).
