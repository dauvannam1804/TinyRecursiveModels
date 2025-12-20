# Kế hoạch Triển khai Tiny Recursive Model (TRM) from Scratch

## 1. Tổng quan Dự án
Mục tiêu là xây dựng một mô hình **Tiny Recursive Model (TRM)** từ đầu (from scratch) để giải quyết các bài toán suy luận (reasoning) dựa trên dataset toán học/logic. Mô hình dựa trên ý tưởng của paper "Recursive Reasoning with Tiny Networks", sử dụng kiến trúc mạng nơ-ron đệ quy (recursive/recurrent) với trọng số chia sẻ (shared weights) để giả lập độ sâu vô hạn mà vẫn giữ số lượng tham số nhỏ.

### Tài nguyên
- **Paper**: Recursive Reasoning with Tiny Networks.pdf
- **Dataset**: `train-00000-of-00001.parquet` (Dạng Problem-Solution).

## 2. Cấu trúc Thư mục (Directory Structure)
Thiết kế theo hướng module hóa, tách biệt giữa cấu hình, dữ liệu, mô hình và huấn luyện để dễ dàng mở rộng và debug.

```
TinyRecursiveModels/
├── configs/                # Chứa các file cấu hình
│   └── default_config.yaml # Config mặc định (model size, lr, steps...)
├── data/                   # Chứa dữ liệu
│   ├── raw/                # File parquet gốc
│   └── processed/          # Tokenizer đã train, cache dataset
├── src/                    # Source code chính
│   ├── __init__.py
│   ├── config.py           # Class quản lý cấu hình (dùng Pydantic hoặc Dataclass)
│   ├── dataset.py          # Logic load và xử lý dữ liệu
│   ├── model.py            # Kiến trúc TRM (Core logic)
│   ├── tokenizer.py        # Logic train và load tokenizer
│   └── trainer.py          # Class quản lý training loop, checkpointing
├── notebooks/              # Jupyter notebooks để EDA, test nhanh
├── scripts/                # Shell scripts để chạy train/eval
│   └── run_train.sh
├── requirements.txt        # Các thư viện cần thiết
└── main.py                 # Entry point của chương trình
```

## 3. Chi tiết Các Module

### 3.1. Cấu hình (`src/config.py`)
Sử dụng `dataclass` để định nghĩa cấu hình, giúp code có gợi ý (type hinting) và dễ quản lý hơn là dùng dictionary trần.
- **Thông số Model**: `d_model`, `n_heads`, `n_layers` (trong 1 block), `n_recurrence` (số lần lặp lại block).
- **Thông số Train**: `batch_size`, `learning_rate`, `max_seq_len`, `deep_supervision_weight`.

### 3.2. Tokenizer (`src/tokenizer.py`)
Do dataset là dạng text toán học/logic đặc thù, ta không nên dùng tokenizer có sẵn của GPT-2/BERT mà nên **train một tokenizer mới** (BPE hoặc Unigram) trên chính dataset này.
- **Thư viện**: `tokenizers` (của HuggingFace) hoặc `sentencepiece`.
- **Hàm chính**: `train_tokenizer(data_path, vocab_size, save_path)`.

### 3.3. Dataset (`src/dataset.py`)
- **Class**: `MathDataset(torch.utils.data.Dataset)`.
- **Xử lý**:
    - Đọc file parquet.
    - Format dữ liệu: `Input: <Problem> \n Output: <Solution> <EOS>`.
    - Tokenize và padding/truncation theo `max_seq_len`.
    - Trả về: `input_ids`, `attention_mask`, `labels`.

### 3.4. Mô hình TRM (`src/model.py`) - **Trọng tâm**
Đây là phần quan trọng nhất, hiện thực hóa ý tưởng của paper.
- **Kiến trúc**:
    - **Embeddings**: Token Embedding + Positional Embedding.
    - **Recursive Block**: Một khối Transformer Encoder/Decoder (ví dụ: gồm 2-4 lớp Self-Attention + FFN). Khối này sẽ được gọi lặp đi lặp lại.
    - **Loop Đệ quy**:
        - Input ban đầu: $H_0 = Embeddings(X)$
        - Lặp $t = 1 \dots T$ (số bước suy luận):
            - $H_t = Block(H_{t-1})$ (Trọng số của Block là không đổi qua các bước).
            - $Logits_t = OutputHead(H_t)$
    - **Deep Supervision**:
        - Thay vì chỉ lấy output ở bước cuối $T$, ta lấy output ở **mỗi bước** $t$.
        - Trong quá trình train, Loss sẽ là tổng hợp của loss tại mỗi bước: $L_{total} = \sum_{t=1}^{T} w_t \cdot L_{CE}(Logits_t, Target)$.
        - Điều này giúp gradient lan truyền tốt hơn và ép mô hình phải đưa ra dự đoán tốt dần lên qua từng bước suy luận.

### 3.5. Training Loop (`src/trainer.py`)
- Tự viết vòng lặp training thay vì dùng `Trainer` có sẵn của HuggingFace để dễ dàng tùy chỉnh phần Deep Supervision Loss.
- **Tính năng**:
    - Gradient Clipping (quan trọng cho mạng đệ quy).
    - Learning Rate Scheduler (Cosine Decay).
    - Logging (WandB hoặc Tensorboard).
    - Lưu checkpoint sau mỗi epoch hoặc số bước nhất định.

## 4. Lộ trình Thực hiện (Step-by-Step Plan)

1.  **Setup & Data Prep**:
    - Tạo cấu trúc thư mục.
    - Viết script `inspect_data.py` để hiểu rõ format (đã làm sơ bộ).
    - Viết `src/tokenizer.py` và chạy train tokenizer trên file parquet.

2.  **Data Pipeline**:
    - Viết `src/dataset.py`.
    - Test thử việc load batch, kiểm tra shape của tensor đầu ra.

3.  **Model Implementation**:
    - Viết `src/model.py`.
    - Bắt đầu với một mô hình nhỏ (Tiny): `d_model=256`, `n_layers=2` (trong block), `n_recurrence=4`.
    - Test forward pass để đảm bảo kích thước tensor đúng và loop hoạt động.

4.  **Training Logic**:
    - Viết `src/trainer.py` với Deep Supervision Loss.
    - Viết `main.py` để kết nối mọi thứ.

5.  **Experiment & Tune**:
    - Chạy train thử trên một phần nhỏ dữ liệu (overfit test) để đảm bảo model học được.
    - Sau đó train full dataset.

## 5. Lưu ý Kỹ thuật (Technical Notes)
- **Gradient Checkpointing**: Nếu gặp vấn đề về bộ nhớ GPU khi tăng số bước lặp ($T$), có thể dùng gradient checkpointing cho mỗi bước lặp.
- **Detached State**: Paper có nhắc đến việc "detach" latent features để giả lập mạng rất sâu mà không tốn bộ nhớ. Có thể thử nghiệm option này: `H_t = Block(H_{t-1}.detach())` (nhưng vẫn giữ gradient cho các phần khác nếu cần). Tuy nhiên, với "Tiny Network", ta nên thử Backprop through time (BPTT) đầy đủ trước.
- **Positional Embeddings**: Cần xem xét có nên cộng lại Positional Embeddings ở mỗi bước lặp hay không. Thường thì chỉ cộng ở đầu vào.

---
**Kết luận**: Kế hoạch này tập trung vào sự rõ ràng và khả năng mở rộng. Việc tách module giúp bạn dễ dàng thay đổi kiến trúc Block hoặc cách tính Loss mà không ảnh hưởng đến toàn bộ hệ thống.
