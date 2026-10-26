# Task 01 — Tự triển khai kiến trúc LLM decoder-only

Mục tiêu: giải thích được từng bước từ văn bản đến token tiếp theo, tự sửa
được từng thành phần và chạy một lần huấn luyện có train/validation loss.

**Chuyển sang máy mới:** làm theo [TASK_01_SETUP.md](TASK_01_SETUP.md) từ bước 1
đến bước 7 để chạy notebook. Khi cần tra cứu sâu, đọc
[TASK_01_FULL_GUIDE.md](TASK_01_FULL_GUIDE.md).
Tài liệu đầy đủ độc lập với output bị Git bỏ qua, có setup macOS/Linux/Windows,
giải thích từng class/function và tạo báo cáo từ các lần chạy trên máy mới.

**Trình bày review bằng notebook:** mở [TASK_01_REVIEW.ipynb](TASK_01_REVIEW.ipynb)
để chạy từng cell và xem code, bảng shape, kết quả, biểu đồ loss cùng một chỗ.
Notebook dùng lại model trong `mini_gpt.py`; kết quả nằm ở `out/notebook_task01/`.

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

Mở terminal trong IDE và chuyển vào `course/`. Từ đây đến cuối bài, giữ
terminal ở thư mục này:

```bash
git rev-parse --show-toplevel
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

Đứng tại root repo chứa `02_llm_internals/` trước khi activate. Đường dẫn Python
phải nằm trong `.venv/bin/` của repo. Khi setup trên máy mới, tạo `.venv` trước
khi activate:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Task này dùng `torch` và `pytest`. Kiểm tra trước:

```bash
python -c "import torch, pytest; print('torch:', torch.__version__); print('pytest:', pytest.__version__)"
```

Nếu thiếu thư viện, cài trong môi trường đang activate:

```bash
python -m pip install torch pytest
```

Môi trường đã kiểm chứng dùng Python 3.11.9. Lab chạy cục bộ, không cần API key.
Cấu hình kiến trúc nằm trong dataclass
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

## 4. Quy trình chạy từ đầu đến cuối

### 4.1. Tạo nơi lưu kết quả và thông tin môi trường

Sau khi hoàn thành mục 2, chạy:

```bash
mkdir -p 02_llm_internals/out/my_task01
set -o pipefail
python -c "import sys, torch, pytest; print(sys.version); print('torch:', torch.__version__); print('pytest:', pytest.__version__)" \
  | tee 02_llm_internals/out/my_task01/environment.txt
```

`tee` vừa hiện output vừa lưu file. `pipefail` giữ mã lỗi của lệnh Python khi
dùng pipeline. Các lần chạy cá nhân lưu ở `my_task01`; bằng chứng do trợ lý
chạy trước đó vẫn ở `task01_review`. Thư mục `out/` được Git bỏ qua.

### 4.2. Kiểm tra dữ liệu và tokenizer ký tự

```bash
python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "02_llm_internals")
from mini_gpt import CharTokenizer

text = Path("02_llm_internals/sample_corpus.txt").read_text()
tokenizer = CharTokenizer(text)
sample = "hello AI"
ids = tokenizer.encode(sample)
print("Số ký tự:", len(text))
print("Vocabulary:", len(tokenizer.chars))
print("Token IDs:", ids)
print("Decode:", tokenizer.decode(ids))
assert tokenizer.decode(ids) == sample
PY
```

Với corpus chưa sửa, phải thấy **2.004 ký tự**, **35 token trong vocabulary**
và decode ra `hello AI`. Corpus minh họa này đủ để kiểm tra cơ chế huấn luyện;
ba câu ngắn trong ảnh không đủ cho context 32 và split 90/10 của lab.

Để thực hành BPE riêng, chạy thêm:

```bash
python 02_llm_internals/bpe.py --merges 50 \
  --text 02_llm_internals/sample_corpus.txt \
  2>&1 | tee 02_llm_internals/out/my_task01/bpe.log
```

### 4.3. Chạy kiểm thử trước khi train

```bash
python -m pytest 02_llm_internals/test_mini_gpt.py -q \
  2>&1 | tee 02_llm_internals/out/my_task01/tests.log
```

Mốc hiện tại là **8 passed**. Nếu có test lỗi, đọc phần `FAILED` và sửa trước
khi đi tiếp. Test bao gồm causal mask, gradient, tokenizer và lưu/nạp checkpoint.

### 4.4. Train 100 bước để kiểm tra toàn bộ luồng

```bash
python -u 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/sample_corpus.txt --device cpu \
  --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 \
  --batch_size 8 --steps 100 --lr 0.003 --dropout 0.1 \
  --log_every 25 --sample_tokens 120 \
  --out 02_llm_internals/out/my_task01/model_100.pt \
  2>&1 | tee 02_llm_internals/out/my_task01/train_100.log
```

`-u` hiện log ngay. Script tự chia 90% đầu corpus cho train và 10% cuối cho
validation; in loss tại bước 0, 25, 50, 75, 100; lưu weights, config và vocabulary;
sau đó sinh thêm 120 token. Seed khởi tạo là 1337.

Kết quả kiểm chứng để đối chiếu (có thể khác nhẹ theo môi trường):

```text
device=cpu chars=2,004 vocab=35 params=0.10M
step     0 | train 3.593 | val 3.591
step   100 | train 2.306 | val 2.347
saved 02_llm_internals/out/my_task01/model_100.pt
=== sample ===
```

Sau lệnh này phải có `model_100.pt` và `train_100.log`. Văn bản mẫu còn lộn
xộn là kết quả quan sát của mô hình nhỏ này; dùng train/val loss và test để
đánh giá việc triển khai, đồng thời ghi rõ hạn chế của chất lượng văn bản.

### 4.5. Train một lần mới với 500 bước để so sánh

```bash
python -u 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/sample_corpus.txt --device cpu \
  --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 \
  --batch_size 8 --steps 500 --lr 0.003 --dropout 0.1 \
  --log_every 25 --sample_tokens 120 \
  --out 02_llm_internals/out/my_task01/model_500.pt \
  2>&1 | tee 02_llm_internals/out/my_task01/train_500.log
```

Đây là lần huấn luyện khởi tạo mới, không tiếp tục từ checkpoint 100 bước.
Đổi số bước cũng đổi thời lượng warmup/cosine của lab. Giữ nguyên các tham số
khác, gồm `log_every`, để dễ đối chiếu.

Trong lần kiểm chứng, train loss cuối là **1.532**, val loss **2.373**. Hãy so
sánh với lần 100 bước và giải thích khoảng cách train/validation; train lâu
hơn không bảo đảm validation tốt hơn trên corpus nhỏ.

### 4.6. Nạp checkpoint để sinh văn bản mà không train lại

Chạy nguyên khối sau sau khi có `model_500.pt`. Lệnh lưu mẫu mới vào log:

```bash
python - <<'PY' | tee 02_llm_internals/out/my_task01/generate.log
import sys
import torch

sys.path.insert(0, "02_llm_internals")
from mini_gpt import Config, GPT

checkpoint = torch.load(
    "02_llm_internals/out/my_task01/model_500.pt",
    map_location="cpu",
    weights_only=True,
)
model = GPT(Config(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])
model.eval()

chars = checkpoint["chars"]
stoi = {char: i for i, char in enumerate(chars)}
prompt = "hello "
inputs = torch.tensor([[stoi[char] for char in prompt]], dtype=torch.long)
torch.manual_seed(42)
output = model.generate(inputs, max_new_tokens=120, temperature=0.8)
print("Config:", checkpoint["config"])
print("Sinh từ prompt:", repr(prompt))
print("".join(chars[i] for i in output[0].tolist()))
PY
```

`eval()` tắt dropout; `generate` đã có `torch.no_grad()`. Vocabulary lấy từ
checkpoint phải khớp weights. Khi đổi prompt, dùng các ký tự đã có trong
`chars`; ký tự mới sẽ gây `KeyError` trong ví dụ kiểm tra trực tiếp này.

### 4.7. Tự thực hành và hoàn thiện báo cáo

Làm phần tự code ở mục 5: viết lại từng hàm theo cách hiểu, chạy test rồi train
lại khi thay đổi mã nguồn. Ghi cấu hình và kết quả thực tế vào ghi chép cá nhân.

Trong `TASK_01_REPORT.md`, điền họ tên, phần đã tự triển khai, môi trường,
loss đo được, mẫu sinh và nhận xét. Giữ rõ nguồn của từng lần chạy. Khi thay
bằng chứng bằng lần chạy của bạn, sửa liên kết log/checkpoint tương ứng sang
`out/my_task01/`. Đọc log đầy đủ trước khi kết luận về overfitting.

```markdown
| Lần chạy | Train loss đầu/cuối | Val loss đầu/cuối | Nhận xét mẫu sinh |
|---|---|---|---|
| 100 bước | ... | ... | ... |
| 500 bước | ... | ... | ... |
```

Bạn đang mở `data/tinyshakespeare.txt`: sau khi hoàn thành hai lần chạy mẫu,
có thể dùng nó bằng cách đổi `--data` và đặt tên checkpoint/log mới. Đây là
thực nghiệm khác với corpus, vocabulary và mốc loss khác; ghi kết quả riêng.

### 4.8. Đóng gói bài nộp sau khi cập nhật báo cáo

Lệnh sau dành cho macOS có `zip`, chạy từ `course/`. Gói gồm mã nguồn,
báo cáo, kết quả cá nhân và bằng chứng kiểm chứng có sẵn mà báo cáo tham chiếu:

```bash
zip -r -FS 02_llm_internals/out/my_task01/submission.zip \
  02_llm_internals/TASK_01_REPORT.md \
  02_llm_internals/TASK_01.md \
  02_llm_internals/LESSON.md \
  02_llm_internals/mini_gpt.py \
  02_llm_internals/bpe.py \
  02_llm_internals/test_mini_gpt.py \
  02_llm_internals/sample_corpus.txt \
  02_llm_internals/out/my_task01 \
  -x '*.zip'
unzip -t 02_llm_internals/out/my_task01/submission.zip
```

Kết quả kiểm tra archive phải báo không có lỗi. `-FS` đồng bộ nội dung archive
với các file được chọn khi đóng gói lại. Gói được tạo từ trạng thái
file tại thời điểm chạy lệnh; nếu sửa báo cáo hoặc chạy lại mô hình, đóng gói
lại trước khi nộp. Sau khi hoàn tất, có thể rời môi trường bằng `deactivate`.

### 4.9. Các lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `No such file or directory` | Kiểm tra `pwd` đang là `course/` và đường dẫn đã gõ đúng |
| `No module named torch` / `pytest` | Activate `.venv`, kiểm tra `sys.executable`, cài bằng `python -m pip` |
| Corpus quá ngắn | Dùng `sample_corpus.txt` hoặc giảm context; cả train và val phải đủ dài |
| `n_embd must be divisible by n_head` | Dùng cặp `64 / 4` như cấu hình bắt đầu |
| `MPS is unavailable` | Dùng `--device cpu` |
| Không tìm thấy checkpoint | Chạy xong bước 4.5, đối chiếu dòng `saved` và đường dẫn bước 4.6 |
| `KeyError` khi đổi prompt | Kiểm tra ký tự có trong vocabulary của checkpoint |

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
