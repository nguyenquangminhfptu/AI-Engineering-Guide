# Task 01 — Tự triển khai kiến trúc LLM decoder-only

Mục tiêu: giải thích được từng bước từ văn bản đến token tiếp theo, tự sửa
được từng thành phần và chạy một lần huấn luyện có train/validation loss.

Để chuẩn bị nộp review, xem [TASK_01_REPORT.md](TASK_01_REPORT.md): đối chiếu
yêu cầu, kiến trúc, kết quả thực nghiệm, bằng chứng kiểm thử và phần người học
cần tự hoàn thiện. Log/checkpoint kiểm chứng nằm trong `out/task01_review/`.

## 1. Sắp xếp file

Giữ mã nguồn trong module hiện tại để có thể đọc lab từ đầu đến cuối:

```text
02_llm_internals/
├── LESSON.md          # Lý thuyết nền tảng
├── TASK_01.md         # Hướng dẫn và tiêu chí hoàn thành task này
├── TASK_01_REPORT.md  # Báo cáo kỹ thuật và bằng chứng cho review
├── bpe.py             # Lab BPE độc lập
├── mini_gpt.py        # Config, tokenizer ký tự, model, train và generate
├── test_mini_gpt.py   # Kiểm tra CLI, checkpoint và sinh văn bản
├── sample_corpus.txt  # Dữ liệu nhỏ, cố định để thực hành
├── data/             # Dữ liệu tải về; Git bỏ qua
└── out/              # Checkpoint và ghi chép kết quả; Git bỏ qua
```

`mini_gpt.py` dùng tokenizer ký tự; `bpe.py` giúp học BPE riêng. Hai tokenizer
chưa được nối với nhau. Các module khác đang nhập `GPT` và `train` từ
`mini_gpt.py`, vì vậy giữ tên file và các hàm đó khi thực hành.

## 2. Môi trường và cấu hình

Chạy lệnh từ `course/`, dùng môi trường `.venv` hiện có:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

Nếu tạo môi trường mới, chỉ cần các lệnh sau cho task này:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

Lab chạy cục bộ, không cần API key. Cấu hình kiến trúc nằm trong dataclass
`Config` của `mini_gpt.py`; tham số dòng lệnh ghi đè mặc định cho từng lần chạy.
Chưa cần thêm file YAML hoặc thư viện quản lý cấu hình.

| Tham số | Giá trị bắt đầu | Ý nghĩa |
|---|---|---|
| `vocab_size` | Tự tính từ corpus | Số ký tự khác nhau |
| `--block_size` | `32` | Số token đầu vào trong mỗi đoạn |
| `--n_embd` | `64` | Số chiều biểu diễn mỗi token |
| `--n_head` | `4` | Số attention head; mỗi head có `64 / 4 = 16` chiều |
| `--n_layer` | `2` | Số Transformer block |
| `--dropout` | `0.1` | Tỷ lệ dropout khi train |
| `--batch_size` | `8` | Số đoạn văn bản trong một batch |
| `--steps` | `100` | Số lần cập nhật trọng số |
| `--lr` | `0.003` | Learning rate cơ sở, có warmup và schedule |
| `--device` | `cpu` | Thiết bị chạy; `auto` tự chọn CUDA/MPS/CPU |

`n_embd` phải chia hết cho `n_head`. Bắt đầu bằng CPU để kiểm tra luồng xử lý
với mô hình nhỏ. Muốn đổi sang MPS trên Mac, dùng `--device mps` khi khả dụng.

## 3. Thứ tự đọc và tự thực hành

1. **Tokenization:** đọc `CharTokenizer` trong `mini_gpt.py`. In kết quả
   encode/decode của `hello AI`. Sau đó chạy `bpe.py` để so sánh cách chia token.
2. **Embedding và vị trí:** đọc `GPT.__init__` và `GPT.forward`. Tìm
   `tok_emb(idx) + pos_emb(pos)`. Đây là position embedding học được.
3. **Masked self-attention:** đọc `CausalSelfAttention`. Theo dõi Q/K/V,
   chia head, tính scores, che tương lai, softmax và trộn V.
4. **Transformer block:** đọc `MLP` và `Block`. Mỗi block gồm attention,
   FFN, LayerNorm và residual connections; khối này được lặp `n_layer` lần.
5. **Đầu ra:** đọc `head`, cross-entropy và `generate`. Khi train, truyền
   logits trực tiếp vào cross-entropy. Khi sinh, dùng softmax để lấy xác suất
   rồi chọn token tiếp theo. `model.eval()` tắt dropout trước khi sinh.

Với cấu hình trên, theo dõi các kích thước:

```text
Token IDs                       [B, T]       = [8, 32]
Token + position embeddings     [B, T, C]    = [8, 32, 64]
Q, K, V sau khi chia head        [B, H, T, D] = [8, 4, 32, 16]
Attention scores                [B, H, T, T] = [8, 4, 32, 32]
Đầu ra mỗi Transformer block    [B, T, C]    = [8, 32, 64]
Logits                          [B, T, V]    = [8, 32, vocab_size]
```

Chuỗi đúng là embedding → position embedding → **N block, mỗi block chứa
attention và FFN** → final LayerNorm → Linear. Không thêm một attention/FFN
riêng trước chồng block.

## 4. Chạy từng bước

Đầu tiên kiểm tra tokenizer:

```bash
python 02_llm_internals/bpe.py --merges 50 \
  --text 02_llm_internals/sample_corpus.txt
```

Sau đó chạy một lần huấn luyện ngắn. Script tự tạo thư mục `out/`:

```bash
python 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/sample_corpus.txt \
  --device cpu \
  --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 \
  --batch_size 8 --steps 100 --lr 0.003 --dropout 0.1 \
  --log_every 25 --sample_tokens 120 \
  --out 02_llm_internals/out/task01_smoke.pt
```

Script dành 90% đầu corpus để train và 10% cuối để validation; in loss ở
bước 0, 25, 50, 75, 100; lưu checkpoint rồi sinh văn bản. Khởi tạo dùng seed
1337. Đây là corpus minh họa với các câu cùng chủ đề, chỉ dùng kiểm tra cơ chế
và luồng huấn luyện. Đánh giá chất lượng ngôn ngữ cần corpus lớn, đa dạng và
tập kiểm tra độc lập. Ba câu ngắn trong ảnh không đủ cho context và split này.

Nếu đã cài dependencies của course (có `pytest`), chạy các kiểm tra offline:

```bash
python -m pytest 02_llm_internals/test_mini_gpt.py -q
```

Để quan sát lâu hơn, đổi `--steps 100` thành `--steps 500` và tên checkpoint
thành `task01_500.pt`. Mỗi lần chỉ thay đổi một tham số để dễ giải thích kết quả.

Ghi kết quả vào `out/task01_notes.md`:

```markdown
| Lần chạy | Config thay đổi | Train loss đầu/cuối | Val loss đầu/cuối | Nhận xét mẫu sinh |
|---|---|---|---|---|
| 01 | Cấu hình bắt đầu | ... | ... | ... |
```

## 5. Tiêu chí hoàn thành và review

- [ ] Encode rồi decode thu lại văn bản ban đầu trong vocabulary.
- [ ] Giải thích được `x` và `y` lệch nhau một token.
- [ ] Vẽ được luồng forward và giải thích các kích thước tensor ở trên.
- [ ] Chỉ ra causal mask và giải thích vì sao phải che token tương lai.
- [ ] Phân biệt position embedding với token embedding.
- [ ] Có checkpoint, train/val loss và mẫu văn bản sinh từ lần chạy thực tế.
- [ ] Giải thích được vì sao train loss thấp chưa chứng minh tổng quát tốt.
- [ ] Review: trình bày kiến trúc, cấu hình, kết quả và một hạn chế trong 5 phút.

Để hoàn thành phần **tự code**, che lần lượt thân hàm `CharTokenizer.encode`,
`CausalSelfAttention.forward`, `MLP.forward`, `Block.forward` và `GPT.forward`,
viết lại từ cách hiểu của bạn rồi chạy lại. Dùng Git diff để đối chiếu với bản
gốc; lưu kết quả từng phần trước khi chuyển sang phần tiếp theo.
