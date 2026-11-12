# GIẢI THÍCH CHI TIẾT LAB 3 — để trả lời khi bị hỏi

Tài liệu này giải thích **tất cả thông số, bảng, biểu đồ, ảnh** được tạo ra và chạy bởi
2 notebook: `src/Lab3_code.ipynb` (Bài thực hành 4 — Wavelet) và `src/Facenet_code.ipynb` (FaceNet realtime).

---

# PHẦN A — LAB3_CODE.IPYNB (So sánh tương đồng ảnh bằng Wavelet)

## A1. Pipeline tổng quát (nói 10 giây khi bị hỏi)

> "Ảnh → chuyển xám + resize 128×128 → biến đổi wavelet db4 mức 3 → lấy subband LL
> → ngưỡng median → hash 484 bit → so khoảng cách Hamming giữa các cặp →
> ngưỡng 0.157 quyết định tương tự/khác → đánh giá accuracy/recall/specificity/ROC."

## A2. Thông số chính của pipeline (CONFIG)

| Thông số | Giá trị | Ý nghĩa | Tại sao chọn |
|---|---|---|---|
| `wavelet` | `db4` | Bộ lọc wavelet Daubechies-4 (dài 8 mẫu) | Trơn, phổ biến, giữ chi tiết tốt hơn Haar |
| `level` | 3 | Số mức phân rã DWT | Level 3 → LL còn 22×22 = 484 hệ số: vừa gọn vừa bền với biến đổi nhẹ |
| `size` | 128 | Kích thước ảnh đầu vào (128×128 xám) | **Bắt buộc cố định** để LL của mọi ảnh cùng kích thước → so được Hamming |
| Hash | 484 bit | 1 bit/hệ số LL (1 nếu hệ số > median, 0 nếu ngược) | Median bất biến khi cả ảnh sáng lên/tối đi cùng lúc |
| `step` | 2 | Bước lượng tử hóa của bản "quantize" (theo slide) | Chỉ dùng để so sánh với pseudocode trong slide |

- **LL (approximation):** phần xấp xỉ — mang "nội dung" ảnh, ít bị nhiễu/ánh sáng/nén ảnh hưởng.
- **HL/LH/HH (detail):** biên ngang/dọc/chéo — nhạy biến đổi, **không dùng** để hash.
- Kích thước LL theo mức (vì hiệu ứng biên của bộ lọc db4 dài 8): level 1 → **67×67**, level 2 → **37×37**, level 3 → **22×22** (từ ảnh 128×128).

💡 **Ví dụ hiểu từng thông số:**
- **`level` đại diện cho "co ảnh xa bao xa":** level 3 = nhìn ảnh như 22×22 → chỉ còn "bố cục to", bỏ chi tiết nhỏ. Đánh đổi thật (số từ bảng khảo sát): cùng bộ db4, **level 1 → hash 4489 bit (acc 0.967)**, **level 3 → hash 484 bit (acc 0.933)** — chênh nhẹ, gọn gần 9 lần.
- **`wavelet='db4'` đại diện cho "kính nhìn trơn":** so với haar (bậc thang, khối vuông), db4 lọc trơn nên biên mới không tạo "vết gợn" trong LL.
- **`size=128` đại diện cho "kích thước khuôn":** mọi ảnh kéo về cùng 1 khuôn trước khi rút gọn — nếu thiếu bước này, LL của ảnh 500px và 300px khác kích thước → **không so được Hamming**.
- **Median trong hash:** LL có 484 hệ số; sắp xếp lấy **giá trị trung vị** (hệ số thứ ~242); mỗi hệ số > median → bit 1. Vì median chỉ so **với nhau trong ảnh** nên cả ảnh sáng lên 30% cũng không đổi thứ tự → hash không đổi (thấy thật: `home_orig` vs `home_bright30` d = **0.0083**).
- **Hamming cụ thể:** 2 hash 484 bit khác nhau 40 bit → khoảng cách = 40/484 ≈ **0.083** → nhỏ hơn ngưỡng 0.157 → kết luận "tương tự".

## A3. Thông số bộ dữ liệu (Task 1)

- **10 ảnh gốc** tải từ bộ mẫu OpenCV: baboon, board, building, butterfly, fruits, home, starry_night, sudoku, messi5, aero1 — 10 nội dung khác nhau, resize về 256×256.
- **5 kiểu biến đổi** (mô phỏng "cùng đối tượng, góc/điều kiện khác nhau"):

| Tên biến đổi | Nghĩa | Làm gì |
|---|---|---|
| `rot12` | rotate 12° | Xoay 12° quanh tâm, giữ kích thước, lấp mép bằng Reflect |
| `scale08` | scale 0.8 | Thu nhỏ còn 80% rồi phóng về 256×256 (mất nét nhẹ) |
| `noise15` | noise σ=15 | Thêm nhiễu Gauss độ lệch chuẩn 15 trên mỗi pixel |
| `bright30` | bright +30% | Tăng độ sáng ~76/255 (`cv2.convertScaleAbs beta=76`) |
| `jpeg30` | JPEG quality 30 | Nén mất dữ liệu mạnh (lưu file .jpg) |

- **Cặp dương (label = 1): 50 cặp** = (ảnh gốc, phiên bản biến đổi của chính nó) — 10 nguồn × 5 biến đổi.
- **Cặp âm (label = 0): 50 cặp** = (ảnh biến đổi của nguồn này, **ảnh gốc của nguồn khác**) — để 2 lớp cân bằng.
- Tổng: **60 ảnh, 100 cặp**, sinh bằng seed 42 → chạy lại cho cùng kết quả (reproducible).
- File mô tả cặp: `data/processed/pairs.csv`, cột: `path_a`, `path_b` (2 ảnh trong cặp), `label` (1 tương tự / 0 khác), `transform` (kiểu biến đổi), `source_a`, `source_b` (nguồn gốc từng ảnh).

💡 **Ví dụ 2 dòng thật trong `pairs.csv` (đọc cho người hỏi nghe):**
```
path_a             path_b           label  transform  source_a    source_b
baboon_orig.png    baboon_rot12.png   1      rot12     baboon.jpg  baboon.jpg   → cùng khỉ, khác góc nhìn
baboon_rot12.png   board_orig.png     0      rot12     baboon.jpg  board.jpg    → khỉ xoay vs mạch điện: khác nội dung
```
- Dòng 1 khoảng cách **0.376** (cặp dương xa nhất của bộ — xoay làm nội dung dịch chỗ); dòng 2 khoảng cách **0.508** → tách lớp đúng.

## A4. Các hàm trong `src/wavelet_hash.py`

| Hàm | Việc gì |
|---|---|
| `to_gray(image, size)` | Ảnh BGR → xám, resize size×size, về float |
| `wavelet_hash(image, wavelet, level, size)` | DWT → lấy LL → `bit = LL > median(LL)` → mảng 1 chiều (484 bit) |
| `wavelet_hash_quantize(...)` | Bản theo slide: lượng tử hóa `round(LL/2)` rồi `bit = giá trị % 2` (không ổn định — dùng để so sánh) |
| `hamming(h1, h2)` | Số bit khác nhau / tổng bit → khoảng cách trong [0,1] |
| `pairwise_distances(hashes, pairs)` | Tính Hamming cho toàn bộ cặp trong `pairs.csv` |
| `evaluate(distances, labels, threshold)` | Đếm TP/TN/FP/FN và tính 5 chỉ số đánh giá |
| `pick_threshold(dist, labels)` | Chọn ngưỡng tối ưu trên **train** theo Youden's J = max(TPR − FPR) |
| `imread_unicode / imwrite_unicode` | Đọc/ghi ảnh bằng `imencode/imdecode + tofile` — vì `cv2.imread` **không đọc được đường dẫn có tiếng Việt** (`Chuyên Ngành`) |

## A5. Giải thích từng ảnh/biểu đồ trong `outputs/`

### `fig_01_dataset_pairs.png` — Xem trước bộ dữ liệu
- 6 dòng × 2 cột: bên trái **cặp TƯƠNG TỰ** (rot12, scale08, noise15), bên phải **cặp KHÁC NHAU** cùng kiểu biến đổi.
- Nhãn bên trái mỗi dòng cho biết: lớp (TƯƠNG TỰ/KHÁC NHAU) + kiểu biến đổi.
- Ý nghĩa: chứng minh bộ dữ liệu hợp lệ — cặp dương là "cùng nội dung, khác điều kiện".

### `fig_02_wavelet_subbands.png` — Subband wavelet của ảnh baboon
- 5 panel: ảnh gốc 128×128, **LL 22×22**, HL (biên ngang), LH (biên dọc), HH (biên chéo).
- Mỗi panel có **colorbar** cho biết giá trị hệ số (LL giữ xấp xỉ, các detail quanh 0).
- Nói khi bị hỏi: "Ta hash trên LL của level 3 vì nó phản ánh nội dung và ổn định; các subband chi tiết nhạy với biến đổi."

### `fig_03_hash_visualize.png` — Hai cách tạo hash
- Panel 1: ảnh gốc baboon. Panel 2: **hash median** (484 bit, vẽ dạng ảnh 22×22). Panel 3: **hash quantize theo slide**.
- Nhận xét: hash median gần như "phân bố ngẫu nhiên ổn định"; bản quantize dải bit dày/trống bất thường → kém phân biệt (thấy rõ ở fig_06).

### `fig_04_distance_hist.png` — Phân bố khoảng cách Hamming (Task 4)
- 2 histogram: **cặp TƯƠNG TỰ** (xanh lá) và **cặp KHÁC NHAU** (đỏ), trục x = khoảng cách Hamming chuẩn hóa [0,1], trục y = số cặp.
- **Số liệu thực tế:**
  - Cặp tương tự: **min 0.000 · mean 0.059 · max 0.376**
  - Cặp khác nhau: **min 0.260 · mean 0.494 · max 0.686**
- Cách đọc: hai dải tách khá rõ, chỉ giao nhau nhỏ ở khoảng 0.26–0.38 → không có ngưỡng 100% chính xác → cần ngưỡng + đánh giá định lượng (fig_05).

### `fig_05_confusion_roc.png` — Đánh giá (Task 5)
Chia **70% train / 30% test** (stratify theo label, seed 42): ngưỡng chọn trên **train**, số đo chấm trên **test**.
- **Panel trái — Confusion matrix (ngưỡng 0.157):** TN=15, FP=0, FN=2, TP=13.
- **Panel phải — ROC trên test:** AUC = **0.996**; chấm đỏ = điểm làm việc của ngưỡng 0.157 (FPR = 0.0, TPR = 0.867).
- **Các chỉ số và công thức (bắt buộc thuộc):**

| Chỉ số | Công thức | Giá trị | Ý nghĩa |
|---|---|---|---|
| Accuracy | (TP+TN)/N | **0.933** | 28/30 cặp phân loại đúng |
| Recall (sensitivity) | TP/(TP+FN) | **0.867** | 13/15 cặp tương tự được nhận đúng |
| **Specificity (độ đặc hiệu)** | TN/(TN+FP) | **1.000** | 15/15 cặp khác nhau được loại đúng |
| Precision | TP/(TP+FP) | **1.000** | 13/13 "tương tự" dự đoán đều đúng |
| F1 | 2PR/(P+R) | **0.929** | Hòa giữa precision & recall |

💡 **Ví dụ hiểu từng ô confusion (nói theo kiểu "dịch sang chuyện đời thường"):**
- **TP = 13** — "dự đoán tương tự và đúng": ví dụ cặp (baboon gốc, baboon nén JPEG) d = **0.004** ≤ 0.157 → đúng là cùng con khỉ.
- **TN = 15** — "dự đoán khác và đúng": ví dụ cặp (baboon thu nhỏ, mạch điện) d = **0.541** > 0.157 → đúng là khác nội dung.
- **FN = 2** — "thực tế tương tự nhưng bị chê": cặp (baboon gốc, baboon xoay 12°) d = **0.376** > 0.157. Đây là cặp dương **xa nhất** của bộ — xoay làm nội dung dịch chỗ → hash lệch. Muốn đỡ FN: hạ ngưỡng (mất specificity).
- **FP = 0** — không có cặp khác nhau nào bị nhầm là tương tự → precision và specificity đều 100%.
- **Đọc nhẩm từng chỉ số:** recall = 13/(13+2) = 0.867; specificity = 15/(15+0) = 1.0; precision = 13/(13+0) = 1.0; accuracy = (13+15)/30 = 0.933.
- 💡 **AUC 0.996 đại diện cho điều gì:** "lấy ngẫu nhiên 1 cặp tương tự và 1 cặp khác nhau, xác suất 99.6% là cặp tương tự có khoảng cách **nhỏ hơn**" — AUC không phụ thuộc ngưỡng, đo chất lượng của *hash*; accuracy phụ thuộc ngưỡng, đo chất lượng của *quyết định*.
- 💡 **Ngưỡng 0.157 (Youden) đại diện cho điều gì:** quét mọi giá trị ngưỡng trên ROC train, lấy điểm **xa đường chéo ngẫu nhiên nhất** (max TPR − FPR). Tại sao không chọn ngưỡng giữa 2 dải cho bằng? Vì Youden ưu tiên cả 2 phía: điểm 0.157 vừa loại sạch nhóm khác (FPR=0) vừa giữ được 13/15 nhóm giống.

- ⚠️ **Lưu ý trả lời:** slide ghi "Độ đặc hiệu: precision" là **nhầm lẫn** — độ đặc hiệu = **specificity** (đo TN rate); precision là phép đo khác (TP/(TP+FP)).

### Bảng sức bền theo biến đổi (trong notebook, Task 5)

| Biến đổi | mean dương | max dương | mean âm | max âm |
|---|---|---|---|---|
| bright30 | 0.018 | 0.050 | 0.494 | 0.674 |
| jpeg30 | 0.007 | 0.017 | 0.489 | 0.686 |
| noise15 | 0.007 | 0.017 | 0.482 | 0.686 |
| **rot12** | **0.261** | **0.376** | 0.467 | 0.591 |
| scale08 | 0.004 | 0.012 | 0.539 | 0.686 |

- Cách nói: "Hash gần như không đổi với sáng/nhiễu/nén/thu nhỏ (mean ~0.004–0.018). **Xoay là biến đổi khó nhất** (mean 0.261) — nội dung dịch chuyển vị trí, nhiều bit đổi dấu; đây là nguồn của 2 cặp FN trong confusion matrix."

### `fig_06_variant_comparison.png` — Khảo sát 18 cấu hình (Task 6 nâng cao)
- Grid: {`median`, `quantize`} × {`haar`, `db4`, `sym4`} × level {1, 2, 3}; mỗi cấu hình hash lại toàn bộ 60 ảnh, chọn ngưỡng trên **cùng bộ train**, đo accuracy trên test + AUC trên 100 cặp.
- 2 bar chart sắp xếp giảm dần: AUC (trái) và test accuracy (phải); đường đứt = mức ngẫu nhiên 0.5.
- **Kết quả thực (file `outputs/variant_results.csv`):**

| # | method | wavelet | level | bits | test_acc | auc |
|---|---|---|---|---|---|---|
| 1 | median | haar | 2 | 1024 | 0.933 | **0.993** |
| 2 | median | db4 | 1 | 4489 | 0.967 | 0.993 |
| 3 | median | haar | 3 | 256 | 0.967 | 0.992 |
| 4 | median | sym4 | 1 | 4489 | 0.967 | 0.992 |
| 5 | median | haar | 1 | 4096 | 0.967 | 0.992 |
| 6 | median | sym4 | 3 | 484 | 0.967 | 0.991 |
| 7 | median | sym4 | 2 | 1369 | 0.967 | 0.990 |
| 8 | median | db4 | 2 | 1369 | 0.933 | 0.988 |
| 9 | median | db4 | 3 | 484 | 0.933 | 0.980 |
| 10–18 | quantize | (tất cả) | 1–3 | 256–4489 | 0.43–0.70 | 0.53–0.74 |

- Cách nói: "**median thắng quantize đậm** ở mọi cấu hình (quantize mất ổn định khi hệ số lướt qua bội số bước lượng tử). Trong nhóm median, 9 cấu hình gần như ngang nhau (AUC 0.98–0.99) — chọn cấu hình là bài toán đánh đổi: **hash càng ngắn càng gọn/nhanh, độ phân biệt giảm nhẹ**. Cấu hình dùng chính trong lab là db4-level3 (484 bit, acc 0.933, AUC 0.980)."

### `fig_07_search_*.png` — Tìm kiếm ảnh theo hash (Task 7 nâng cao)
- CSDL = hash của **10 ảnh gốc**. 2 truy vấn: `baboon_rot12` (ảnh xoay) và `butterfly_jpeg30` (ảnh nén).
- Mỗi hình: truy vấn (trái) + **top 5** kết quả xếp theo khoảng cách Hamming tăng dần.
- **Kết quả thực:** cả 2 truy vấn trả đúng ảnh **#1**.
- Cách nói: "Vì mỗi ảnh chỉ 484 bit, so 1 truy vấn với hàng triệu ảnh là phép XOR popcount cực nhanh; hash dùng để **lọc ứng viên**, bước 2 xác nhận lại bằng SIFT/ORB hoặc CNN."

## A6. Ghi chú kỹ thuật so với pseudocode trong slide (điểm cộng khi bị hỏi)

1. `pywt.wavedec2` trả **tuple lồng nhau** (LL là mảng, detail là các tuple) — slide flatten trực tiếp là lỗi.
2. Phải **resize cố định 128×128** trước, nếu không LL khác kích thước → Hamming không so được.
3. `pywt.quantize` **không có** trong PyWavelets hiện hành → tự viết `np.round(coeff / step)`.

---

# PHẦN B — FACENET_CODE.IPYNB (Nhận diện khuôn mặt realtime)

## B1. Pipeline (nói 10 giây)

> "Webcam OpenCV → MTCNN phát hiện khuôn mặt → crop 160×160 → FaceNet
> (InceptionResnetV1 pretrained 'vggface2') → embedding 512 chiều đã chuẩn hóa
> → cosine similarity với các khuôn mặt đã đăng ký → **> 0.7: Matched, < 0.7: Unknown**."

## B2. Thông số chính

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| `THRESHOLD` | **0.7** | Ngưỡng của đề bài: similarity > 0.7 → "Matched", ngược lại "Unknown" |
| `device` | `cuda` nếu có, không thì `cpu` | Tính toán trên GPU/CPU tự động |
| `MTCNN(keep_all=True)` | — | Phát hiện **tất cả** khuôn mặt trong frame (không chỉ 1) |
| Kích thước crop | 160×160 | FaceNet vggface2 yêu cầu đầu vào cố định 160×160 |
| Embedding | **512 chiều**, chuẩn hóa L2 (norm = 1) | Khuôn mặt giống nhau → vector gần nhau; chuẩn hóa để cosine chỉ phụ thuộc góc |
| `n_shots` | 10 ảnh/người | Đăng ký: chụp 10 lần rồi **lấy trung bình embedding** → giảm nhiễu |
| Chỉ số so sánh | **Cosine similarity** ∈ [−1, 1] | Chọn người có similarity cao nhất trong CSDL |

## B3. Các hàm chính

| Hàm | Việc gì |
|---|---|
| `open_webcam(idx)` | Mở camera (`CAP_DSHOW` — khởi động nhanh trên Windows); assert nếu không mở được |
| `detect_faces_bgr(frame)` | `mtcnn.detect` trên frame đã đổi BGR→RGB; trả [(box, prob), ...]; **prob là mảng object do interop torch/numpy** → ép float |
| `embed_faces(frame, boxes)` | Crop từng box → resize 160×160 → RGB → tensor [0,1] → FaceNet → (N, 512) |
| `register_face(name, n_shots)` | Vòng lặp chụp: nhấn **`s`** để chụp (đếm x/n_shots trên màn hình), **`q`** hủy; lưu ảnh + trả embedding trung bình |
| `match_face(vec, db)` | Cosine similarity giữa vec và toàn bộ CSDL → (tên, sim) cao nhất |

## B4. File được tạo ra khi chạy FaceNet

| File | Sinh khi nào | Nội dung |
|---|---|---|
| `data/raw/faces/<tên>_<n>.jpg` | Mỗi lần nhấn `s` trong `register_face` | **Frame chụp** (ảnh full khung hình camera) |
| `data/raw/faces/faces_db.pt` | Ô "Lưu CSDL" sau khi đăng ký xong | Dictionary `{tên: tensor 512 chiều}` — phiên sau chạy ô "nạp CSDL" là nhận diện được **không cần chụp lại** |

## B5. Các nhãn hiển thị khi chạy realtime

- Khung **xanh lá** + `Matched: <tên> (0.xx)` khi sim > 0.7
- Khung **đỏ** + `Unknown (0.xx)` khi sim ≤ 0.7
- Không phát hiện khuôn mặt: chữ "Khong phat hien khuon mat" (đỏ)
- Nhấn `q` để thoát, `s` để chụp khi đăng ký

## B6. Trả lời nhanh những câu hay bị hỏi

1. **Vì sao FaceNet không cần huấn luyện?** — Lấy weights đã train sẵn trên ~3 triệu ảnh khuôn mặt (vggface2); chỉ dùng để **trích đặc trưng**, phần so khớp là phép tính cosine.
2. **Vì sao chọn 0.7?** — Là ngưỡng mặc định của đề bài; thực tế nên tinh chỉnh theo bộ test riêng (mờ, nghiêng, che khuất → hạ ngưỡng có thể tăng recall nhưng tăng nhầm).
3. **Vì sao đăng ký 10 ảnh rồi lấy trung bình?** — Mỗi frame có nhiễu; trung bình 10 embedding ổn định hơn 1 ảnh (kỹ thuật enrollment tiêu chuẩn).
4. **Hạn chế?** — Nhạy độ phân giải thấp/góc nghiêng lớn/che khuất; người ngoài CSDL luôn Unknown; chưa có chống giả mạo (ảnh in, video replay).

---

# PHẦN C — SỐ LIỆU NHANH (thuộc lòng để trả lời)

- Bộ dữ liệu: **60 ảnh — 100 cặp (50/50)**, seed 42.
- Hash: **484 bit/ảnh** (LL 22×22, db4, level 3, ảnh 128×128).
- Phân bố Hamming: dương **0.000–0.376** (mean 0.059) — âm **0.260–0.686** (mean 0.494).
- Ngưỡng (Youden trên train): **0.157**.
- Kết quả test (30 cặp): TP 13, TN 15, FP 0, FN 2 → **acc 0.933, recall 0.867, specificity 1.000, precision 1.000, F1 0.929, AUC 0.996**.
- Khảo sát 18 cấu hình: **median > quantize**; top AUC 0.993 (median-haar-L2); cấu hình chính của lab (median-db4-L3): acc 0.933, AUC 0.980.
- Tìm kiếm: 2 truy vấn (xoay, JPEG) đều trả đúng **#1**.
- FaceNet: embedding **512**, ngưỡng **0.7**, crop **160×160**, đăng ký **10 ảnh/người**.
