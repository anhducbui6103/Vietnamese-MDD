# Vietnamese MDD Colab

Project này dùng cho bài toán Vietnamese Mispronunciation Detection and Diagnosis (MDD) trong MDD Challenge 2026. Workflow chính được tách thành 2 phần rõ ràng:

- `MDD_Training.ipynb`: train model và lưu checkpoint.
- `MDD_Prediction.ipynb`: load checkpoint, predict public/private test và tạo file nộp bài.

## 1. Chuẩn Bị Data

Tải dữ liệu từ trang cuộc thi:

```txt
https://aihub.ml/competitions/1001#participate-get-data
```

Sau khi tải và giải nén, đặt dữ liệu vào folder `data/` với cấu trúc:

```txt
data/
  MDD-Challenge-2025-training-set/
    audio_data/
      train/
    metadata/
      train.csv
      train_phones.csv
      lexicon_vmd.txt

  MDD-Challenge-2025-public-test/
    audio_data/
      public_test/
    metadata/
      public_test.csv
      public_test_phones.csv

  MDD-Challenge-2025-private-test/
    audio_data/
      private_test/
    metadata/
      private_test_submission.csv
      private_test_submission_example.csv
      README_submission.md
```

Khi chạy trên Google Colab, upload cả project lên Google Drive, ví dụ:

```txt
/content/drive/MyDrive/SinhThanh/
  MDD_Training.ipynb
  MDD_Prediction.ipynb
  data/
```

## 2. Mở Notebook Trên Google Colab

Có 2 cách chạy:

1. Upload project lên Google Drive, sau đó mở notebook bằng Colab.
2. Push project lên GitHub, sau đó mở notebook từ GitHub trong Colab.

Khuyến nghị dùng GPU:

```txt
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

## 3. Mount Google Drive

Trong mỗi notebook, mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Đặt biến root project:

```python
BASE = "/content/drive/MyDrive/SinhThanh"
```

Nếu folder trên Drive của bạn khác tên, chỉ cần sửa `BASE`.

## 4. Training Workflow

Dùng notebook:

```txt
MDD_Training.ipynb
```

Input chính:

```python
TRAIN_ROOT = f"{BASE}/data/MDD-Challenge-2025-training-set"
PUBLIC_ROOT = f"{BASE}/data/MDD-Challenge-2025-public-test"
WORK_DIR = f"{BASE}/MDD_Work"
```

Metadata training:

```python
TRAIN_META = f"{TRAIN_ROOT}/metadata/train_phones.csv"
PUBLIC_META = f"{PUBLIC_ROOT}/metadata/public_test_phones.csv"
```

Training notebook thực hiện các bước:

1. Cài dependencies trên Colab.
2. Load metadata và kiểm tra audio file.
3. Tạo vocabulary từ phone transcript.
4. Chia training set thành train/dev split.
5. Tạo dataset và dataloader cho audio + canonical phones.
6. Khởi tạo model `AcousticLinguisticMDD`.
7. Train model với CTC loss.
8. Evaluate trên dev set.
9. Lưu checkpoint tốt nhất.

Output dự kiến:

```txt
/content/drive/MyDrive/SinhThanh/MDD_Work/
  vocab.json
  train_split.csv
  dev_split.csv
  public_gt.csv
  history_new_account.csv
  checkpoints_new_account/
    best_mdd_vietnamese.pt
    last_mdd_vietnamese.pt
```

Checkpoint quan trọng nhất để dùng cho prediction:

```txt
/content/drive/MyDrive/SinhThanh/MDD_Work/checkpoints_new_account/best_mdd_vietnamese.pt
```

## 5. Điều Chỉnh Cấu Hình Training

Trong `MDD_Training.ipynb`, các tham số nên được gom tại một cell config để dễ thay đổi:

```python
PRETRAINED_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-5
EVAL_START_EPOCH = 2
SEED = 42
```

Ý nghĩa:

- `PRETRAINED_MODEL`: backbone audio pretrained model.
- `BATCH_SIZE`: batch size khi train. Giảm nếu bị out-of-memory.
- `EPOCHS`: số epoch training.
- `LR`: learning rate.
- `EVAL_START_EPOCH`: epoch bắt đầu evaluate và save best checkpoint.
- `SEED`: random seed để chia train/dev ổn định hơn.

Danh sách pretrained model được phép sử dụng:

```txt
facebook/wav2vec2-base-100h
nguyenvulebinh/wav2vec2-base-vietnamese-250h
facebook/hubert-base-ls960
```

Không dùng pretrained model khác ngoài 3 model trên.

Ví dụ đổi sang HuBERT:

```python
PRETRAINED_MODEL = "facebook/hubert-base-ls960"
```

Ví dụ cấu hình debug nhanh:

```python
BATCH_SIZE = 2
EPOCHS = 1
LR = 1e-5
```

Ví dụ cấu hình train dài hơn:

```python
BATCH_SIZE = 4
EPOCHS = 20
LR = 5e-6
```

## 6. Prediction Workflow

Dùng notebook:

```txt
MDD_Prediction.ipynb
```

Prediction notebook nên dùng các path:

```python
BASE_DIR = "/content/drive/MyDrive/SinhThanh"

PRIVATE_ROOT = f"{BASE_DIR}/data/MDD-Challenge-2025-private-test"
PRIVATE_CSV = f"{PRIVATE_ROOT}/metadata/private_test_submission.csv"

CKPT_PATH = f"{BASE_DIR}/MDD_Work/checkpoints_new_account/best_mdd_vietnamese.pt"
OUT_DIR = f"{BASE_DIR}/MDD_Work/prediction"
```

Prediction workflow:

1. Mount Google Drive.
2. Cài dependencies giống training notebook.
3. Load vocabulary và khởi tạo model.
4. Load checkpoint `best_mdd_vietnamese.pt`.
5. Load public/private test audio.
6. Chạy inference.
7. Tạo file kết quả.
8. Zip file submission nếu cuộc thi yêu cầu.

Output dự kiến:

```txt
/content/drive/MyDrive/SinhThanh/MDD_Work/prediction/
  results.csv
  prediction.zip
```

Nếu cần tạo file theo format khác, hãy dựa theo file mẫu:

```txt
data/MDD-Challenge-2025-private-test/metadata/private_test_submission_example.csv
```

## 7. Mô Hình

Model chính là `AcousticLinguisticMDD`, kết hợp:

- audio encoder từ pretrained wav2vec2/HubERT;
- phone encoder cho chuỗi canonical phones;
- cross-attention giữa audio representation và canonical phone representation;
- classifier để dự đoán transcript phones;
- CTC loss để học alignment audio-phone.

Ý tưởng chính: model không chỉ nghe audio, mà còn biết câu phone đúng mong đợi (`canonical`) để phát hiện và dự đoán phần phát âm thực tế (`transcript`).

## 8. Metric

Notebook tính các metric:

- `PER`: Phone Error Rate.
- `F1`: F1 cho phát hiện lỗi phát âm.
- `DER`: Diagnosis Error Rate.
- `Score`: điểm tổng hợp.

Công thức score trong notebook:

```python
Score = 0.5 * F1 + 0.4 * (1 - DER) + 0.1 * (1 - PER)
```

## 9. Lưu Ý Khi Chạy

- Nên chạy trên Colab GPU. Chạy CPU sẽ rất chậm.
- Sau cell cài dependencies, Colab có thể yêu cầu restart runtime. Hãy restart rồi chạy lại từ đầu.
- Nếu gặp lỗi path, kiểm tra lại `BASE`, `TRAIN_ROOT`, `PUBLIC_ROOT`, `PRIVATE_ROOT`.
- Nếu gặp CUDA out-of-memory, giảm `BATCH_SIZE`.
- `data/` có thể rất lớn, không nên commit toàn bộ audio lên Git nếu không cần.
- File checkpoint và output prediction nên lưu trong Google Drive để không mất khi Colab disconnect.

## 10. Cấu Trúc Project

```txt
Vietnamese-MDD-Colab/
  README.md
  MDD_Training.ipynb
  MDD_Prediction.ipynb
  data/
    MDD-Challenge-2025-training-set/
    MDD-Challenge-2025-public-test/
    MDD-Challenge-2025-private-test/
```
