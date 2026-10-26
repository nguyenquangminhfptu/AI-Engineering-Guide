# Lab 1 / Task 01 — Hướng dẫn đầy đủ từ clone repo đến review mini GPT

**Phạm vi:** task kiến trúc LLM trong module `02_llm_internals`, gồm tokenization,
embedding, vị trí, masked self-attention, Transformer block, pretraining và inference.
“Lab 1” trong tài liệu này là bài kiến trúc bạn đang làm, khác module `01_agents`.

Tài liệu dành cho máy mới: bắt đầu chỉ với source code clone từ Git, tự tạo môi
trường, tự train và thu thập bằng chứng để báo cáo. Các class/function dưới đây
được giải thích theo implementation hiện có, không yêu cầu framework huấn luyện khác.

Nếu muốn bắt đầu bằng một checklist ngắn theo đúng thứ tự thao tác, mở
[TASK_01_SETUP.md](TASK_01_SETUP.md), rồi quay lại đây để tra cứu từng hàm.

Để trình bày code và kết quả trực tiếp khi review, dùng
[TASK_01_REVIEW.ipynb](TASK_01_REVIEW.ipynb). Notebook gọi lại `mini_gpt.py`, chạy
hai ngân sách, vẽ loss, demo reload và tạo bản nháp báo cáo. Đọc tài liệu này
để hiểu implementation; mục 3.6 hướng dẫn mở notebook.

## Mục lục và cách học

1. [Repo nào cần push và clone?](#1-repo-nào-cần-push-và-clone)
2. [Cấu trúc file và vai trò](#2-cấu-trúc-file-và-vai-trò)
3. [Cài môi trường trên máy mới](#3-cài-môi-trường-trên-máy-mới)
4. [Thư viện nào thực hiện việc gì?](#4-thư-viện-nào-thực-hiện-việc-gì)
5. [Chạy lần đầu và lưu bằng chứng](#5-chạy-lần-đầu-và-lưu-bằng-chứng)
6. [Token, batch, logits và mục tiêu học](#6-token-batch-logits-và-mục-tiêu-học)
7. [Cấu hình và số tham số](#7-cấu-hình-và-số-tham-số)
8. [Toàn bộ `mini_gpt.py`](#8-toàn-bộ-mini_gptpy)
9. [Toàn bộ `bpe.py`](#9-toàn-bộ-bpepy)
10. [Toàn bộ `test_mini_gpt.py`](#10-toàn-bộ-test_mini_gptpy)
11. [Nạp model và sinh văn bản](#11-nạp-model-và-sinh-văn-bản)
12. [Đọc kết quả và làm thực nghiệm](#12-đọc-kết-quả-và-làm-thực-nghiệm)
13. [Tạo báo cáo của máy mới](#13-tạo-báo-cáo-của-máy-mới)
14. [Đóng gói bài nộp](#14-đóng-gói-bài-nộp)
15. [Lỗi thường gặp](#15-lỗi-thường-gặp)
16. [Câu hỏi review và checklist](#16-câu-hỏi-review-và-checklist)
17. [Tài liệu chính thức](#17-tài-liệu-chính-thức)

Lượt 1: làm mục 1–5 để có một lần chạy thành công. Lượt 2: đọc mục 6–10 cùng
source code và tự giải thích từng phép biến đổi. Lượt 3: làm mục 11–16, sửa
từng phần theo cách hiểu rồi đo lại. Đọc đến đâu chạy và ghi nhận đến đó.

## 1. Repo nào cần push và clone?

### 1.1. Trên máy hiện tại

Git repo nằm trong `course/`, không nằm ở thư mục workspace cha `AI Engineering/`.
Trong terminal đang ở `course/`, kiểm tra:

```bash
git rev-parse --show-toplevel
git status --short
```

Đọc diff và commit các tài liệu cần mang sang máy mới. Ví dụ cho lần bổ sung này:

```bash
git diff
git add .gitignore README.md 02_llm_internals/TASK_01.md 02_llm_internals/TASK_01_FULL_GUIDE.md 02_llm_internals/TASK_01_REVIEW.ipynb
git commit -m "Document complete mini GPT lab and fresh machine setup"
git push
```

Lệnh `git push` dùng remote/branch bạn đã cấu hình. Nếu hosting yêu cầu lệnh
thiết lập remote lần đầu, dùng URL repo của bạn và hướng dẫn của hosting đó.
Các lệnh này để bạn thực hiện; việc viết tài liệu không tự commit hoặc push.

### 1.2. Những gì Git mang sang máy mới

| Nội dung | Được Git lưu? | Máy mới làm gì? |
|---|---|---|
| Source `.py`, test, lesson và các hướng dẫn `.md` | Có khi đã commit | Clone về để đọc/chạy |
| `sample_corpus.txt` | Có | Dùng ngay làm corpus mẫu |
| `.venv/` | Không | Tạo lại trên máy mới |
| `02_llm_internals/data/` | Không | Tải/tạo corpus khi cần |
| `02_llm_internals/out/` | Không | Tạo lại log và báo cáo thực nghiệm |
| File `*.pt` | Không | Train lại để có checkpoint mới |
| `.env` | Không | Task này không cần API key |

Vì vậy sau clone, không thấy `train_500.log` hoặc `model_500.pt` cũ là bình
thường. Các con số trong `TASK_01_REPORT.md` là kết quả tham khảo từ máy cũ,
không tự trở thành kết quả của máy mới. Báo cáo cũ có liên kết tới output bị
Git bỏ qua; bạn sẽ tạo bằng chứng mới theo mục 5 và 13.

Thư mục sách `LLMs-from-scratch-main/` và hướng dẫn agent ở workspace cha
cũng không thuộc repo `course/`; lab này không phụ thuộc vào chúng.

### 1.3. Clone trên máy mới

Cài Git trước nếu máy chưa có. Thay chuỗi URL mẫu bằng URL repo thật:

```bash
git clone "REPLACE_WITH_YOUR_REPOSITORY_URL" ai-engineering-course
cd ai-engineering-course
git status
git rev-parse --show-toplevel
```

**Từ đây tất cả lệnh chạy ở root repo vừa clone**, nơi có `README.md`,
`requirements.txt`, `01_agents/`, `02_llm_internals/`… Không chạy thêm
`cd course` nếu bạn đã clone chính repo này vào `ai-engineering-course`.

## 2. Cấu trúc file và vai trò

```text
ai-engineering-course/                # Root Git sau clone; tên có thể khác
├── README.md
├── requirements.txt                  # Dependencies cho toàn bộ course
├── .gitignore
├── 01_agents/                        # Lab agent dùng LLM API, khác bài này
├── 02_llm_internals/
│   ├── LESSON.md                     # Nền tảng lý thuyết
│   ├── TASK_01.md                    # Hướng dẫn thực hành ngắn hơn
│   ├── TASK_01_FULL_GUIDE.md          # Tài liệu đang đọc
│   ├── TASK_01_REVIEW.ipynb          # Notebook học và trình bày review
│   ├── TASK_01_REPORT.md              # Báo cáo tham khảo của lần kiểm chứng cũ
│   ├── mini_gpt.py                   # Tokenizer ký tự + model + train + generate
│   ├── bpe.py                        # Lab BPE độc lập
│   ├── test_mini_gpt.py              # Test hành vi mã nguồn
│   ├── sample_corpus.txt             # Corpus nhỏ được commit
│   ├── data/                        # Tạo sau clone, Git bỏ qua
│   └── out/                         # Tạo sau clone, Git bỏ qua
│       └── my_task01/
│           ├── environment.json
│           ├── packages.txt
│           ├── tokenizer.log
│           ├── bpe.log
│           ├── tests.log
│           ├── train_100.log
│           ├── train_500.log
│           ├── model_100.pt
│           ├── model_500.pt
│           ├── generate.log
│           ├── metadata.json
│           └── REPORT.md             # Báo cáo thực nghiệm trên máy mới
├── 03_datasets/
├── 04_finetuning/                    # Có lab import GPT/train từ mini_gpt.py
├── 05_evals/
├── 06_rag_serving/
└── common/
```

Phân biệt ba việc:

- **Train model:** `mini_gpt.py` đọc văn bản qua `--data`, cập nhật trọng số.
- **Train tokenizer BPE:** `bpe.py` học luật gộp byte; chưa được nối vào GPT này.
- **Test code:** `pytest` chạy `test_mini_gpt.py` để kiểm tra tính đúng đắn.

File `.txt` là nội dung để học; file `test_*.py` là chương trình kiểm thử.
Test end-to-end có train thử một bước trên corpus tạm, nhưng đó không phải
dataset hoặc checkpoint của thực nghiệm chính.

### Chỉ làm bài qua notebook: cần mang theo những file nào?

Giữ nguyên thư mục `02_llm_internals/` với năm file chạy sau:

1. `TASK_01_REVIEW.ipynb` — mở, chạy từng cell và trình bày review.
2. `mini_gpt.py` — implementation mà notebook import.
3. `test_mini_gpt.py` — notebook gọi pytest kiểm tra source.
4. `sample_corpus.txt` — corpus mặc định của notebook, đã được Git lưu.
5. `bpe.py` — notebook hiện đọc hash file này khi tạo metadata, vì vậy vẫn
   cần giữ file dù bạn chưa chạy lab BPE riêng.

Mang thêm `TASK_01_FULL_GUIDE.md` để tra cứu. Các file `TASK_01.md`,
`TASK_01_REPORT.md`, `LESSON.md` là tài liệu bổ sung; chưa cần mở chúng để chạy
notebook. Các module `01_agents`, `03_datasets`… không cần cho lần thực hành này.
Bạn có thể clone cả repo rồi chỉ làm trong module 02; không cần xóa các phần khác.

Không cần chuyển `.venv`, `data/tinyshakespeare.txt` hoặc output/checkpoint cũ.
Notebook mặc định đọc `sample_corpus.txt` và tạo `out/notebook_task01/` khi chạy.
Output đã lưu trong `.ipynb` cũng được mang theo khi commit notebook; chọn
Restart Kernel and Run All để thay bằng kết quả thực tế của máy mới.

## 3. Cài môi trường trên máy mới

### 3.1. Chọn Python và thiết bị

Dùng **Python 3.11** làm mốc gần môi trường đã kiểm chứng. Cài Python và Git
bằng bộ cài phù hợp hệ điều hành; xác nhận phiên bản trước khi tạo `.venv`.
Với Ubuntu/Debian, nếu báo thiếu `venv`, cài gói `python3-venv` tương ứng
Python đang dùng, rồi tạo lại môi trường. [Python venv](https://docs.python.org/3.11/library/venv.html)
giải thích việc tạo môi trường riêng và activation theo từng shell.

Bắt đầu bằng **CPU**. Với model khoảng 104 nghìn tham số, CPU đủ để kiểm tra
toàn bộ luồng. Sau khi chạy ổn mới thử MPS/CUDA và ghi thành một thực nghiệm khác.

### 3.2. macOS/Linux — Bash hoặc Zsh

Đang ở root repo:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
export PYTHONUTF8=1
python -m pip install --upgrade pip
```

Nếu `python3` trỏ tới phiên bản khác, dùng `python3.11 -m venv .venv` khi Python
3.11 đã được cài. `PYTHONUTF8` giúp nhất quán khi đọc corpus tiếng Việt.

Trên macOS, cài hai dependency trực tiếp của bài:

```bash
python -m pip install torch pytest
```

Trên Linux, để dùng bản PyTorch CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
```

Lựa chọn wheel CPU/CUDA/ROCm phải phù hợp hệ điều hành và phần cứng; tham khảo
[bộ chọn cài đặt PyTorch](https://pytorch.org/get-started/locally/) và
[các lệnh wheel chính thức](https://pytorch.org/get-started/previous-versions/).
Các lệnh trên chỉ cài `torch`, không cần torchvision/torchaudio cho code của lab.

### 3.3. Windows — PowerShell

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
```

Nếu activation script bị chính sách của máy chặn, có thể dùng trực tiếp
Python trong venv, ví dụ `.\.venv\Scripts\python.exe -m pip install pytest`,
và thay `python` trong các lệnh tiếp theo bằng đường dẫn đó. Không cần đổi
chính sách toàn máy để chạy lab. Nếu `py` chưa được nhận, kiểm tra bộ cài Python.

### 3.4. Kiểm tra đúng môi trường

Lệnh này dùng được trên cả ba hệ điều hành sau khi activate:

```bash
python -c "import sys, torch, pytest; print(sys.executable); print(sys.version); print('torch', torch.__version__); print('pytest', pytest.__version__); print('CUDA', torch.cuda.is_available()); print('MPS', torch.backends.mps.is_available())"
```

Python executable phải nằm trong `.venv/bin/` hoặc `.venv\Scripts\` của repo.
CUDA/MPS báo `False` vẫn chạy bài bằng `--device cpu`.

Task này chỉ cần cài trực tiếp **torch + pytest**; pip tự xử lý dependency
chuyển tiếp của chúng. `requirements.txt` ở root còn chứa SDK Gemini, MCP,
fine-tuning, RAG… cho các module khác. Có thể cài cả file đó khi học toàn course;
riêng bài hiện tại không dùng các SDK ấy hoặc API key.

Phiên bản PyTorch/pytest trong báo cáo cũ là quan sát của môi trường cũ, không
phải yêu cầu bắt buộc phải tìm đúng wheel trên hệ điều hành mới. Ghi lại phiên
bản thực tế, chạy test và đối chiếu kết quả trên máy mới.

### 3.5. Quy tắc copy lệnh theo shell

- Lệnh **một dòng** `python ...` dùng được trong Bash/Zsh và PowerShell.
- Khối nhiều dòng nối bằng `\` và `python - <<'PY'` bên dưới dành cho Bash/Zsh.
- Với PowerShell, ghép lệnh CLI nhiều dòng thành một dòng; không copy ký tự `\`
  dùng nối dòng của Bash. Phần train có lệnh PowerShell riêng ở mục 5.5.
- Với các khối `python - <<'PY'`, copy **phần Python giữa hai dòng mở/đóng** vào
  file `scratch_task01.py` ở root repo, rồi chạy `python scratch_task01.py`.
  Dùng lại file nháp cho từng đoạn; đây chỉ là cách nhập lệnh trên Windows.
- Khi xong, chuyển/xóa file nháp khỏi source bạn định commit. Hướng dẫn có thể
  chạy bằng các đoạn Python rời, không cần thêm runner hoặc tách model thành package.

### 3.6. Cách dễ theo dõi khi báo cáo: notebook

Sau khi activate `.venv` và cài `torch pytest`, thêm công cụ notebook và biểu đồ:

```bash
python -m pip install jupyterlab matplotlib
python -m jupyterlab 02_llm_internals/TASK_01_REVIEW.ipynb
```

Mở notebook, chọn kernel thuộc `.venv`, điền `AUTHOR`, dùng **Restart Kernel and
Run All**, rồi lưu file để giữ output. Các cell đã có giải thích, kiểm tra
tokenizer/shape/mask, test, train 100/500, biểu đồ và reload. Xem
[cài đặt JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

Notebook lưu artifacts tại `out/notebook_task01/`, khác `out/my_task01/` của
quy trình CLI bên dưới. Không trộn hai thư mục khi tổng hợp. Bản notebook được
commit có thể chứa output kiểm chứng cũ; chạy lại trên máy báo cáo để có số
liệu thực tế. Bổ sung phần phân tích của bạn vào notebook và `REPORT.md` được
tạo. Mô hình vẫn nằm trong `mini_gpt.py`, không có một implementation GPT khác.

## 4. Thư viện nào thực hiện việc gì?

### 4.1. Hai package cần cài trực tiếp

| Package | Nơi dùng | Vai trò |
|---|---|---|
| `torch` | Model và test | Tensor, layer, autograd, optimizer, save/load |
| `pytest` | `test_mini_gpt.py` | Chạy test, fixture, parameterization |

`import torch.nn as nn` là namespace các layer trong PyTorch.
`import torch.nn.functional as F` là các phép tính như `gelu`, `softmax`,
`cross_entropy`. Không phải hai package pip khác nhau.

### 4.2. Thư viện chuẩn Python — không pip install riêng

| Import | Nhiệm vụ trong source hoặc các đoạn thu thập bằng chứng |
|---|---|
| `argparse` | Đọc `--steps`, `--data`, `--n_embd`… từ terminal |
| `math` | `sqrt`, `log`, `cos`, `pi` |
| `time` | Đo thời gian trong training loop |
| `urllib.request` | Tải Tiny Shakespeare khi không truyền `--data` |
| `dataclasses.dataclass` | Sinh constructor cho `Config` |
| `pathlib.Path` | Xử lý file và đường dẫn |
| `collections.Counter` | Đếm cặp token trong BPE |
| `sys` | Import module từ thư mục đánh số, thông tin Python |
| `json`, `hashlib`, `platform`, `datetime` | Ghi cấu hình, fingerprint và môi trường |
| `re` | Trích loss từ log |
| `zipfile` | Đóng gói bài nộp |

Tokenizer ký tự và BPE đều tự viết. Không dùng tokenizer có sẵn từ một model
pretrained. PyTorch cung cấp phép toán/layer và vi phân tự động; kiến trúc,
mask, cách chia head và vòng train được viết trong source của bài.

## 5. Chạy lần đầu và lưu bằng chứng

### 5.1. Tạo thư mục và ghi nhận môi trường

Các đoạn Python sau chạy được từ file nháp trên PowerShell như mục 3.5.
Khối này tạo output, ghi thời gian có timezone, phiên bản và commit hiện tại:

```bash
python - <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pytest
import torch

def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

folder = Path("02_llm_internals/out/my_task01")
folder.mkdir(parents=True, exist_ok=True)
corpus = Path("02_llm_internals/sample_corpus.txt")
info = {
    "recorded_at": datetime.now().astimezone().isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "pytest": pytest.__version__,
    "cpu_threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
    "mps_available": torch.backends.mps.is_available(),
    "git_commit": git_value("rev-parse", "HEAD"),
    "git_status": git_value("status", "--short"),
    "dataset": str(corpus),
    "dataset_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
}
(folder / "environment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False, indent=2))
PY
```

Lưu danh sách phiên bản package trong venv mới:

```bash
python -m pip freeze > 02_llm_internals/out/my_task01/packages.txt
```

`packages.txt` là snapshot môi trường thực nghiệm, chưa phải lockfile dùng
chung mọi OS. Commit SHA xác định source đã commit; phần `git_status` ghi nhận
thay đổi chưa commit. Tạo lại thông tin này khi source/môi trường thay đổi.

`git_value` chỉ đọc thông tin Git. Khi chạy từ ZIP không có `.git`, hoặc chưa
cài Git trên máy chỉ giải nén để xem, các field Git nhận `null`; thông tin
phiên bản và hash corpus vẫn được ghi. Nếu clone repo, các field phải phản ánh
commit/status thật của bạn.

### 5.2. Kiểm tra corpus và tokenizer dùng cho model

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "02_llm_internals")
from mini_gpt import CharTokenizer

text = Path("02_llm_internals/sample_corpus.txt").read_text(encoding="utf-8")
tok = CharTokenizer(text)
sample = "hello AI"
ids = tok.encode(sample)
result = f"characters={len(text)}\nvocabulary={len(tok.chars)}\nids={ids}\ndecoded={tok.decode(ids)}\n"
assert tok.decode(ids) == sample
Path("02_llm_internals/out/my_task01/tokenizer.log").write_text(result, encoding="utf-8")
print(result)
PY
```

Corpus chưa sửa có 2.004 ký tự và vocabulary 35 ký tự. IDs của `hello AI` là
`[17, 14, 20, 20, 23, 1, 4, 7]`. Đổi corpus có thể đổi toàn bộ mapping ID.

### 5.3. Chạy test và lab BPE

Bash/Zsh:

```bash
set -o pipefail
python -m pytest 02_llm_internals/test_mini_gpt.py -q 2>&1 | tee 02_llm_internals/out/my_task01/tests.log
python 02_llm_internals/bpe.py --merges 50 --text 02_llm_internals/sample_corpus.txt 2>&1 | tee 02_llm_internals/out/my_task01/bpe.log
```

PowerShell:

```powershell
python -m pytest 02_llm_internals/test_mini_gpt.py -q 2>&1 | Tee-Object -FilePath 02_llm_internals/out/my_task01/tests.log
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
python 02_llm_internals/bpe.py --merges 50 --text 02_llm_internals/sample_corpus.txt 2>&1 | Tee-Object -FilePath 02_llm_internals/out/my_task01/bpe.log
if ($LASTEXITCODE -ne 0) { throw "BPE failed" }
```

Test hiện tại phải có **8 passed**. Nếu fail, xử lý trước khi train chính.
`tee`/`Tee-Object` hiển thị đồng thời lưu log. BPE in merges và cách tách mẫu;
đó là thí nghiệm tokenizer riêng, không sửa tokenizer của GPT.

### 5.4. Train trên Bash/Zsh

Chạy từ root repo, sau bước tạo output:

```bash
python -u 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/sample_corpus.txt --device cpu \
  --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 \
  --batch_size 8 --steps 100 --lr 0.003 --dropout 0.1 \
  --log_every 25 --sample_tokens 120 \
  --out 02_llm_internals/out/my_task01/model_100.pt \
  2>&1 | tee 02_llm_internals/out/my_task01/train_100.log
```

Sau khi hoàn thành, chạy thực nghiệm thứ hai:

```bash
python -u 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/sample_corpus.txt --device cpu \
  --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 \
  --batch_size 8 --steps 500 --lr 0.003 --dropout 0.1 \
  --log_every 25 --sample_tokens 120 \
  --out 02_llm_internals/out/my_task01/model_500.pt \
  2>&1 | tee 02_llm_internals/out/my_task01/train_500.log
```

`-u` hiện log ngay. Lần 500 bước khởi tạo mới, không tiếp tục checkpoint 100.
Giữ nguyên cấu hình, corpus và `log_every` để dễ đối chiếu; lịch learning rate
cũng phụ thuộc `steps`, nên hai ngân sách huấn luyện có schedule khác nhau.

### 5.5. Cùng hai lệnh train trên PowerShell

Copy mỗi lệnh thành một dòng:

```powershell
python -u 02_llm_internals/mini_gpt.py --data 02_llm_internals/sample_corpus.txt --device cpu --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 --batch_size 8 --steps 100 --lr 0.003 --dropout 0.1 --log_every 25 --sample_tokens 120 --out 02_llm_internals/out/my_task01/model_100.pt 2>&1 | Tee-Object -FilePath 02_llm_internals/out/my_task01/train_100.log
if ($LASTEXITCODE -ne 0) { throw "Training 100 failed" }
python -u 02_llm_internals/mini_gpt.py --data 02_llm_internals/sample_corpus.txt --device cpu --n_layer 2 --n_head 4 --n_embd 64 --block_size 32 --batch_size 8 --steps 500 --lr 0.003 --dropout 0.1 --log_every 25 --sample_tokens 120 --out 02_llm_internals/out/my_task01/model_500.pt 2>&1 | Tee-Object -FilePath 02_llm_internals/out/my_task01/train_500.log
if ($LASTEXITCODE -ne 0) { throw "Training 500 failed" }
```

Nếu dùng Windows PowerShell cũ, file do `Tee-Object` ghi có thể là UTF-16;
đoạn tổng hợp kết quả ở mục 13 xử lý cả UTF-8 và UTF-16.
Xem [quy tắc encoding của PowerShell](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding?view=powershell-7.5).

### 5.6. Nhận biết hoàn thành

Một lệnh thành công in `device=...`, các dòng `step ... train ... val ...`,
`saved ...model_100.pt` hoặc `model_500.pt`, rồi `=== sample ===` và văn bản.
Các output này phải được tạo từ lần chạy của chính máy mới.

Nếu chưa đủ 500 bước hoặc có traceback, chưa dùng log đó như kết quả hoàn tất.
Các lần thử mới nên đổi tên hoặc chuyển output cũ sang một thư mục riêng để
không lẫn cấu hình. Các đoạn tổng hợp bên dưới dùng hai tên `model_100.pt` và
`model_500.pt` của cấu hình mốc.

## 6. Token, batch, logits và mục tiêu học

### 6.1. Ký hiệu dùng xuyên suốt

| Ký hiệu | Ý nghĩa | Cấu hình mốc |
|---|---|---:|
| `B` | Batch size, số đoạn trong một bước | 8 |
| `T` | Số token mỗi đoạn | 32 |
| `C` | Embedding dimension | 64 |
| `H` | Số attention head | 4 |
| `D` | Chiều mỗi head, `C // H` | 16 |
| `V` | Vocabulary size | 35 |
| `N` | Số Transformer block | 2 |

Một tensor là mảng số có các chiều. Token IDs là số nguyên `torch.long`;
embeddings, logits và trọng số là số thực. ID chỉ là chỉ mục, không thể hiện
token “lớn hơn” hay “nhỏ hơn” về ngữ nghĩa.

### 6.2. Một ví dụ next-token prediction

Với chuỗi gốc `hello ` và context 5:

```text
x = h e l l o
y = e l l o ␣
```

Vị trí 0 dùng `h` dự đoán `e`; vị trí 1 dùng prefix `he` dự đoán `l`… Causal
mask làm cho việc tính song song toàn chuỗi khi train không cho mỗi vị trí
nhìn thấy đáp án tương lai. Attention vẫn được nhìn token hiện tại.

Logits có shape `[B, T, V]`: tại mỗi vị trí có một điểm số cho từng token có
thể đi tiếp. Trong cấu hình mốc là `[8, 32, 35]`.

### 6.3. Loss và cập nhật trọng số

Với một mục tiêu đúng, cross-entropy tương ứng `-ln(p_token_đúng)`. Xác suất
đúng càng cao, loss càng thấp. Đoán đều giữa 35 token có loss `ln(35) ≈ 3.555`.
Mô hình khởi tạo nhỏ nên loss ban đầu gần mốc đó, không buộc phải bằng đúng.

PyTorch nhận logits chưa chuẩn hóa và token ID mục tiêu; lab flatten thành
`[B*T, V]` và `[B*T]`. Không softmax thủ công trước `F.cross_entropy`.
[CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
mô tả giao diện logits/target này.

Backpropagation tính gradient của loss theo trọng số. AdamW dùng gradient
và trạng thái optimizer để cập nhật trọng số; bản thân lời gọi `backward()`
chưa cập nhật chúng. Số tham số không tăng khi tăng số bước train.

### 6.4. Step, epoch và validation

Một step ở đây xử lý `8 * 32 = 256` vị trí dự đoán. 100 step tương ứng 25.600
mục tiêu, có lặp do lấy cửa sổ ngẫu nhiên. `steps=100` không phải 100 epoch;
script không duyệt từng mẫu một lần theo epoch.

Train split gồm 1.803 ký tự, val split 201 ký tự với corpus mốc. Validation
không cập nhật trọng số, nhưng các cửa sổ đo có thể chồng lấn; không có test
set độc lập. File unit test Python cũng không phải test set chất lượng model.

## 7. Cấu hình và số tham số

### 7.1. Mặc định của source khác cấu hình mốc

| CLI / giá trị | Mặc định khi không truyền | Cấu hình hướng dẫn |
|---|---:|---:|
| `--n_layer` | 4 | 2 |
| `--n_head` | 4 | 4 |
| `--n_embd` | 128 | 64 |
| `--block_size` | 128 | 32 |
| `--batch_size` | 32 | 8 |
| `--steps` | 1500 | 100 hoặc 500 |
| `--lr` | 0.003 | 0.003 |
| `--dropout` | 0.1 | 0.1 |
| `--device` | auto | cpu |
| `--log_every` | 200 | 25 |
| `--sample_tokens` | 400 | 120 |
| `--data` | Tải/cache Tiny Shakespeare, có fallback | `sample_corpus.txt` |
| `--out` | `mini_gpt.pt` cạnh source | `out/my_task01/model_*.pt` |

`vocab_size` trong `Config` có mặc định 65, nhưng `main()` ghi đè bằng số ký
tự của corpus. `--lr` là learning rate cơ sở; scheduler làm learning rate hiệu
dụng thay đổi theo step. Tham số CLI không được lưu tất cả trong `Config`:
checkpoint chỉ lưu cấu hình kiến trúc; cần giữ log/metadata cho batch, steps, lr…

Ràng buộc: các kích thước, steps, batch và log interval dương; `n_embd % n_head == 0`;
`0 <= dropout < 1`; lr dương; số token sinh không âm. Mỗi split cần ít nhất
`block_size + 2` ký tự theo kiểm tra và sampler hiện có.

### 7.2. Model mốc có chính xác bao nhiêu tham số?

| Thành phần | Cách tính | Số tham số |
|---|---|---:|
| Token embedding | `35 * 64` | 2.240 |
| Position embedding | `32 * 64` | 2.048 |
| QKV của một block | `64 * 192 + 192` | 12.480 |
| Attention output projection | `64 * 64 + 64` | 4.160 |
| FFN up | `64 * 256 + 256` | 16.640 |
| FFN down | `256 * 64 + 64` | 16.448 |
| Hai LayerNorm trong một block | `2 * (64 + 64)` | 256 |
| Tổng một block | Cộng năm hàng thành phần block | 49.984 |
| Hai block | `2 * 49.984` | 99.968 |
| Final LayerNorm | `64 + 64` | 128 |
| Output head | Dùng chung trọng số token embedding | Không cộng thêm |
| **Tổng duy nhất** | `2.240 + 2.048 + 99.968 + 128` | **104.384** |

Head và token embedding dùng cùng `Parameter`, nên không tính hai lần.
`model.parameters()` của PyTorch loại trùng tham số chia sẻ. Causal mask là
buffer, không phải tham số học; checkpoint còn chứa buffer/metadata nên dung
lượng `.pt` không bằng chính xác `số_tham_số * 4` byte.

Tăng `n_embd` làm nhiều ma trận tăng gần theo bình phương; tăng `n_layer` thêm
block. Đổi vocabulary hoặc context cũng thay số tham số embedding. Với `C`
cố định, đổi số head hợp lệ không đổi số tham số của implementation QKV này;
nó đổi cách chia chiều. Train 500 bước vẫn là 104.384 tham số như 100 bước.

## 8. Toàn bộ `mini_gpt.py`

Đọc source song song với phần này. Thứ tự thực thi chương trình:

```text
main
 ├─ parse CLI + kiểm tra tham số
 ├─ seed + chọn device
 ├─ load_text → CharTokenizer → tensor → train/val split
 ├─ Config → GPT
 ├─ train → estimate_loss / get_batch / GPT.forward / backward
 ├─ torch.save
 └─ model.eval → GPT.generate → decode → print
```

### 8.1. Imports, `HERE` và `SHAKESPEARE`

`HERE = Path(__file__).resolve().parent` là thư mục chứa `mini_gpt.py`, không
phụ thuộc terminal đang đứng đâu. `SHAKESPEARE` là URL corpus trong repo
`karpathy/char-rnn`. Cache và output mặc định dùng `HERE`; đường dẫn bạn truyền
qua `--data`/`--out` tương đối lại tính từ working directory. Đây là lý do phải
chạy các lệnh hướng dẫn từ root repo.

### 8.2. `Config`

`@dataclass` tạo constructor và biểu diễn cấu hình. Ví dụ
`Config(vocab_size=35, n_embd=64, ...)` là một đối tượng giữ các hyperparameter
kiến trúc, chưa tạo layer hoặc huấn luyện. Các field là `vocab_size`,
`block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`.

`cfg.__dict__` chuyển các field thành dict để lưu. `Config(**checkpoint["config"])`
dùng dict đó dựng lại cùng kiến trúc. Kiểm tra dữ liệu/CLI nằm trong `main()`;
tự gọi `Config` không tự thực hiện toàn bộ kiểm tra của CLI.

### 8.3. `CausalSelfAttention.__init__(c)`

Nhận cấu hình và dựng các module:

- `super().__init__()` khởi tạo cơ chế quản lý module/parameter của PyTorch.
- `self.n_head` lưu số head.
- `qkv = nn.Linear(C, 3*C)` tạo ba projection Q, K, V trong một layer để tính gộp.
- `proj = nn.Linear(C, C)` trộn kết quả sau khi ghép các head.
- `Dropout` bỏ ngẫu nhiên một số giá trị khi train, tắt trong eval.
- `torch.tril(torch.ones(Tmax, Tmax))` tạo tam giác dưới gồm 1, phần tương lai là 0.
- `.view(1, 1, Tmax, Tmax)` thêm hai chiều để broadcast qua batch và head.
- `register_buffer("mask", mask)` cho mask đi theo `.to(device)` và `state_dict`,
  nhưng không coi nó là trọng số để optimizer cập nhật.

Ở đây “causal” nghĩa là prediction của prefix không phụ thuộc token phía sau.
`block_size` quyết định kích thước mask tối đa.

### 8.4. `CausalSelfAttention.forward(x)` — từng phép biến đổi

Đầu vào và đầu ra đều `[B, T, C]`; trong mốc là `[8, 32, 64]`.

1. `B, T, C = x.shape`: đọc kích thước thực tế, `T` có thể nhỏ hơn context tối đa.
2. `self.qkv(x)`: `[B,T,C] → [B,T,3*C]`, mốc `[8,32,192]`.
3. `.split(C, dim=2)`: tách Q, K, V, mỗi tensor `[B,T,C]`.
4. `.view(B,T,H,C//H).transpose(1,2)`: mỗi tensor thành `[B,H,T,D]`,
   mốc `[8,4,32,16]`. Head có projection học được riêng trong ma trận gộp.
5. `q @ k.transpose(-2,-1)`: dot product mỗi query với mọi key trong head,
   thành `[B,H,T,T]`. Hàng là vị trí query, cột là vị trí key được xem.
6. Chia `sqrt(D)`: kiểm soát độ lớn scores theo số chiều mỗi head.
7. `.masked_fill(mask == 0, -inf)`: đặt điểm tương lai thành âm vô cực.
8. `softmax(..., dim=-1)`: xác suất/trọng số chú ý trên các key cho mỗi query;
   tương lai có trọng số 0. Trước dropout, mỗi hàng có tổng 1.
9. Dropout trên weights: khi train, tổng sau dropout không buộc vẫn bằng 1.
10. `att @ v`: trộn các value, về `[B,H,T,D]`.
11. `.transpose(1,2).contiguous().view(B,T,C)`: ghép head. `contiguous()` bảo
    đảm bố trí bộ nhớ phù hợp cho `view` sau phép transpose.
12. `proj` rồi dropout: trả về `[B,T,C]` cho residual connection.

Q/K/V là ba phép chiếu tuyến tính học được từ cùng input; không phải ba chuỗi
văn bản độc lập. Nhiều head có khả năng học các quan hệ khác nhau, nhưng không
có nhãn cố định như “head 1 luôn học ngữ pháp”.

Ma trận scores có T×T phần tử trên mỗi head, nên phần attention này tăng chi
phí theo bình phương chiều dài context. Với T=32 và model nhỏ, baseline rất
nhẹ; tăng context lên nhiều lần không chỉ tăng chi phí tuyến tính.

Ví dụ mask với T=4:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Không dùng `0` làm score tương lai: `softmax(0)` vẫn cho xác suất dương.
Mask đặt `-inf` trước softmax. Mỗi hàng có ít nhất vị trí hiện tại nên không
phải toàn `-inf` trong input hợp lệ.

### 8.5. `MLP.__init__(c)` và `MLP.forward(x)`

Constructor tạo `up: C → 4*C`, `down: 4*C → C` và dropout. Forward là:

```python
self.drop(self.down(F.gelu(self.up(x))))
```

Mốc shape: `[8,32,64] → [8,32,256] → GELU → [8,32,64]`. GELU tạo phi tuyến;
nếu chỉ ghép các layer tuyến tính mà không có phi tuyến thì khả năng biểu
diễn bị hạn chế. FFN biến đổi từng vị trí với cùng trọng số, không trực tiếp
trộn giữa các vị trí; attention đảm nhiệm việc trao đổi thông tin giữa token.

### 8.6. `Block.__init__(c)` và `Block.forward(x)`

Constructor tạo `ln1`, `attn`, `ln2`, `mlp`. `LayerNorm(C)` chuẩn hóa trên chiều
đặc trưng cuối của mỗi token, có scale/bias học được. Forward:

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

Đây là **Pre-LayerNorm**: norm trước module. Residual cộng đầu vào với biến
đổi cùng shape, giúp giữ luồng thông tin/gradient qua chồng block. Dòng thứ
hai dùng `x` đã cập nhật của dòng đầu. Attention và FFN nằm trong **mỗi block**;
chồng N block lặp cả hai thành phần, không thêm một attention/FFN ngoài chồng đó.

### 8.7. `GPT.__init__(c)`

Ghép toàn bộ kiến trúc:

| Thuộc tính | Ý nghĩa |
|---|---|
| `config` | Giữ cấu hình cho train/generate |
| `tok_emb = Embedding(V,C)` | Token ID → vector C chiều |
| `pos_emb = Embedding(Tmax,C)` | Vị trí 0…Tmax-1 → vector học được |
| `drop` | Dropout sau tổng embedding |
| `blocks = ModuleList(...)` | N block được PyTorch đăng ký quản lý |
| `ln_f` | LayerNorm cuối |
| `head = Linear(C,V,bias=False)` | Biểu diễn cuối → logits của vocabulary |

`ModuleList` đăng ký các block để xuất hiện trong `model.parameters()`,
`state_dict()` và `.to(device)`. `head.weight = tok_emb.weight` chia sẻ cùng
trọng số đầu vào/đầu ra. `self.apply(self._init_weights)` áp dụng khởi tạo
cho các module con.

### 8.8. `GPT._init_weights(m)`

Đây là static method, nhận một module `m`. Nếu là `Linear` hoặc `Embedding`,
khởi tạo weight bằng phân phối normal có mean 0, std 0,02. Bias của Linear,
nếu có, được đặt 0. LayerNorm giữ khởi tạo riêng của PyTorch.

Mục tiêu là giữ logits ban đầu ở mức hợp lý, đặc biệt khi embedding chia sẻ
trọng số với head. Comment trong source về loss rất lớn khi bỏ khởi tạo là
hiện tượng của cấu hình/lần chạy cụ thể, không phải hằng số đảm bảo luôn bằng 80.

### 8.9. `GPT.forward(idx, targets=None)`

`idx` là token IDs `[B,T]`. `targets`, khi có, cũng `[B,T]` và chứa token kế tiếp.

1. `arange(T, device=idx.device)` tạo vị trí `[T]` trên cùng thiết bị.
2. `tok_emb(idx)` tạo `[B,T,C]`; `pos_emb(pos)` tạo `[T,C]` và broadcast qua B.
3. Cộng hai embedding, rồi dropout.
4. Duyệt các block, giữ shape `[B,T,C]`.
5. Final LayerNorm và head tạo `[B,T,V]` logits.
6. Nếu có target, flatten logits và target, tính cross-entropy trung bình.
7. Trả về `(logits, loss)`; loss là `None` khi không truyền target.

Gọi `model(x,y)` sử dụng cơ chế `nn.Module` để gọi `forward`. Embedding vị trí
ở đây học được và dùng vị trí tuyệt đối trong cửa sổ, khác sinusoidal/RoPE.
Đầu vào dài hơn `block_size` không phù hợp bảng vị trí/mask; `generate` tự crop.

### 8.10. `GPT.generate(idx, max_new_tokens, temperature=0.8, top_k=None)`

Nhận prompt đã encode `[B,Tprompt]`, trả lại prompt cộng các token mới.
Decorator `@torch.no_grad()` tắt dựng graph gradient cho đoạn sinh.

Mỗi vòng:

1. Crop input về `block_size` token cuối để forward không vượt context.
2. Gọi model không target, nhận logits cho mọi vị trí trong cửa sổ.
3. Lấy `logits[:, -1, :]`: vị trí cuối dự đoán token ngay sau prompt hiện tại.
4. Chia temperature, với temperature phải **lớn hơn 0** trong hàm này.
5. Nếu có `top_k`, giữ các logits trong nhóm cao nhất; phần còn lại thành `-inf`.
6. Softmax tạo xác suất; `torch.multinomial(...,1)` lấy mẫu một token.
7. `torch.cat` nối token vào chuỗi; lặp lại đến đủ `max_new_tokens`.

Hàm chưa có EOS hoặc dừng theo nội dung, không có top-p, không có KV cache.
Nó forward lại cửa sổ ở mỗi token. Khi crop cửa sổ, vị trí được đánh lại từ 0.
Nếu dùng `top_k` qua Python, chọn từ 1 đến V; CLI hiện không có flag top-k hoặc
temperature. Temperature thấp làm phân phối sắc hơn nhưng không bảo đảm câu đúng.
Giá trị 0 sẽ chia cho 0; muốn greedy cần tự dùng `argmax` thay vì truyền 0.

`model.eval()` và `no_grad()` làm hai việc khác nhau: eval đổi hành vi dropout;
no_grad tắt ghi graph. Hàm generate có no_grad nhưng không tự gọi eval;
`main()` và đoạn inference hướng dẫn phải gọi eval trước.
[Autograd notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html) giải
thích sự khác nhau giữa chế độ gradient và eval.

### 8.11. `CharTokenizer.__init__(text)`

`sorted(set(text))` lấy danh sách ký tự khác nhau và sắp xếp để mapping có thứ
tự cố định với cùng corpus. `stoi` ánh xạ character → ID bằng `enumerate`.
Khoảng trắng, dấu câu và newline cũng là token. Trong Python, mỗi phần tử
chuỗi là một Unicode code point; chữ có dấu dạng tổ hợp có thể gồm nhiều phần tử.

Đây là việc dựng vocabulary từ dữ liệu, không có gradient hoặc trọng số neural.
Không dùng quy tắc “một từ = một token”.

### 8.12. `CharTokenizer.encode(s)`

Trả list ID bằng lookup `stoi[c]`. Điều kiện `if c in self.stoi` khiến ký tự
ngoài vocabulary **bị bỏ qua**. Đây là giới hạn của demo; dùng `encode` với
prompt mới cần kiểm tra độ phủ ký tự để không làm mất nội dung âm thầm.

### 8.13. `CharTokenizer.decode(ids)`

Lookup `chars[i]`, rồi nối thành chuỗi. Với IDs hợp lệ và chuỗi chỉ chứa ký tự
đã biết, `decode(encode(s)) == s`. ID ngoài khoảng sẽ gây lỗi; không có UNK,
BOS hoặc EOS được thêm tự động.

### 8.14. `load_text(path=None)`

- Có `path`: đọc file đó, không gọi API và không tải Shakespeare.
- Không có path: tìm `HERE/data/tinyshakespeare.txt`.
- Chưa có cache: tạo `data/`, tải corpus qua `urllib.request.urlretrieve`.
- Tải lỗi: in nguyên nhân, ghép các `LESSON.md` làm fallback.
- Có cache: đọc lại cache.

Fallback làm dataset thay đổi, vì vậy luôn xem output và ghi hash corpus.
Dùng `--data sample_corpus.txt` như hướng dẫn để thực nghiệm mốc độc lập mạng.
`read_text()` trong source dùng encoding mặc định của Python; UTF-8 mode ở
phần setup giúp khi corpus có tiếng Việt. Các đoạn lưu bằng chứng ở đây chỉ
định UTF-8 rõ ràng.

### 8.15. `get_batch(data, block_size, batch_size, device)`

Nhận tensor ID 1 chiều của **một split**. `torch.randint` chọn vị trí bắt đầu
ngẫu nhiên; `stack` ghép các cửa sổ thành batch. Mỗi input lấy T token, target
lấy T token bắt đầu muộn hơn một vị trí. Cuối cùng `.to(device)` chuyển batch
sang CPU/MPS/CUDA. Split đủ dài là điều kiện trước khi gọi hàm.

Implementation lấy chỉ số với cận `len(data) - block_size - 1`, cận trên là
exclusive. Cận này bỏ qua cửa sổ hợp lệ cuối cùng so với sampler tận dụng toàn
bộ cửa sổ; đây là chi tiết hiện tại của lab, không làm thay đổi nguyên tắc
next-token shift. `main` yêu cầu mỗi split có ít nhất `T+2` token tương ứng.

### 8.16. `estimate_loss(model, splits, block_size, batch_size, device, iters=20)`

`splits` là dict như `{"train": train_data, "val": val_data}`. Hàm:

1. Có `@torch.no_grad()`, nên không dựng gradient graph.
2. Gọi eval để tắt dropout cho phép đo.
3. Lấy 20 batch ngẫu nhiên cho mỗi split, tính loss và trung bình.
4. Trả dict `{"train": float, "val": float}`.
5. Gọi `model.train()` trước khi trả để training loop tiếp tục.

Đây là **ước lượng** qua batch, chưa phải loss chính xác trên mọi token của
split. Hàm đặt lại train mode vô điều kiện, không khôi phục mode ban đầu;
nó được thiết kế để gọi trong vòng train của lab.

Lấy batch validation cũng tiêu thụ RNG. Vì dùng chung RNG, đổi `log_every`
hoặc thêm các lệnh random có thể làm đổi batch train sau đó dù seed ban đầu
giống nhau. Muốn thí nghiệm chặt hơn cần tách generator/lịch đánh giá cố định.

### 8.17. `pick_device()`

Ưu tiên `cuda` nếu PyTorch thấy CUDA; tiếp theo `mps` nếu khả dụng trên Mac;
cuối cùng `cpu`. `--device cpu` bỏ qua cơ chế auto này. Cài PyTorch CPU không
tự có hỗ trợ CUDA chỉ vì máy có GPU; cần wheel/driver phù hợp nếu muốn chạy CUDA.

### 8.18. `train(...)` — optimizer, schedule, đo loss và backward

Đối số gồm model, train/val tensors, số bước, lr, batch size, device,
`log_every=200`, và `params=None`. `params` cho phép module 04 chỉ train một
nhóm trọng số; mặc định dùng toàn bộ `model.parameters()`.

Phần chuẩn bị:

- `AdamW(..., weight_decay=0.01)` tạo optimizer và trạng thái cập nhật.
- `warmup = max(1, steps // 20)` lấy khoảng 5% số bước làm mốc warmup.
- `LambdaLR` nhân lr cơ sở với hệ số nhỏ nhất giữa nhánh tăng tuyến tính và
  đường cosine giảm tới khoảng 10%. Code này tính cosine theo `s/steps`,
  không phải hai pha hoàn toàn tách biệt “xong warmup mới bắt đầu cosine”.
- Với 100 bước, warmup=5 nên lr hiệu dụng lúc khởi tạo scheduler là 0,0006.
  Với 500 bước, warmup=25 nên mốc đầu là 0,00012.

Vòng `range(steps + 1)` có cả bước 0 và bước cuối để đo loss. Khi
`step == steps`, nó đo rồi break **trước update**, nên có đúng `steps` lần
cập nhật, không phải `steps+1`.

Mỗi bước cập nhật:

```python
x, y = get_batch(...)
_, loss = model(x, y)
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
sched.step()
```

`zero_grad` bỏ gradient tích lũy từ vòng trước. `backward` tính đạo hàm.
Gradient clipping giới hạn norm tổng, giúp tránh update quá lớn bất thường.
`opt.step` sửa trọng số. `sched.step` sửa learning rate cho bước tiếp theo.
Không dùng gradient accumulation trong lab này.

Log có step, train loss, val loss, lr và thời gian từ lúc bắt đầu vòng train.
Thời gian trong log không bao gồm toàn bộ import/setup/sampling; tổng thời
gian chạy lệnh có thể lớn hơn. Hàm chỉ in log, không trả history hoặc tự lưu
best checkpoint. `tee`/`Tee-Object` mới là thứ lưu stdout ra file thực nghiệm.

### 8.19. `main()` — lắp toàn bộ chương trình

Đây là entry point của CLI:

1. Tạo parser, khai báo flags và defaults như bảng mục 7.
2. Kiểm tra số chiều, divisibility, dropout/lr/sample count và thiết bị yêu cầu.
3. `torch.manual_seed(1337)` cố định RNG khởi tạo/lấy mẫu trong môi trường này.
4. Chọn device, đọc văn bản, dựng tokenizer từ toàn corpus.
5. Encode thành tensor `long`, chia 90/10 theo vị trí, kiểm tra split đủ dài.
6. Dựng `Config` từ CLI với vocabulary thực tế; tạo `GPT(...).to(device)`.
7. In số ký tự, vocab, số tham số và mốc `ln(vocab)`.
8. Gọi `train`.
9. Tạo parent directory và `torch.save` dict weights/config/chars vào `--out`.
10. Gọi eval, chọn token đầu newline nếu có hoặc ID 0, sinh và decode mẫu.

Vocabulary biết tập ký tự của cả train và validation, nhưng weights không
được train bằng val targets. Với đánh giá nghiêm ngặt hơn, cần thiết kế
tokenizer/dataset split và chính sách ký tự lạ rõ ràng.

Checkpoint hiện **không lưu** optimizer state, scheduler state, step và RNG
state. Dùng để nạp inference được; để resume chính xác toàn bộ training cần
mở rộng phần save/load. CLI chưa có `--resume`.

Cuối file, `if __name__ == "__main__": main()` làm cho chạy script thì train,
nhưng `import mini_gpt` để dùng class/test không tự gọi `main`.

## 9. Toàn bộ `bpe.py`

### 9.1. BPE khác tokenizer ký tự như thế nào?

`CharTokenizer` cho mỗi code point một ID. BPE trong lab bắt đầu bằng **256
byte token**, rồi gộp các cặp thường gặp thành token mới. Một token BPE có thể
đại diện nhiều byte/ký tự; không nhất thiết là cả từ. Training BPE chỉ đếm và
học luật merge, không có neural network, backpropagation hoặc AdamW.

Ví dụ `abab`, bắt đầu từ bytes `[97,98,97,98]`. Cặp `(97,98)` xuất hiện hai
lần, được gộp thành ID 256; chuỗi thành `[256,256]`. Vocabulary có thêm token
256 tương ứng `b"ab"`. Đây là ví dụ thuật toán, chưa phải GPT học sinh văn bản.

### 9.2. `pair_counts(ids)`

`zip(ids, ids[1:])` tạo các cặp liền nhau; `Counter` trả dict-like chứa tần
suất. Với `[97,98,97,98]`: `(97,98):2`, `(98,97):1`. Chuỗi ít hơn hai phần tử
cho Counter rỗng. Việc đếm có thể tính các cặp chồng lấn; bước merge áp dụng
tham lam từ trái sang phải với các lần gộp không chồng nhau.

### 9.3. `merge(ids, pair, new_id)`

Duyệt `i` từ trái sang phải, tích lũy `out`. Nếu `ids[i:i+2]` khớp pair, ghi
`new_id` và nhảy hai vị trí; nếu không, giữ token hiện tại và nhảy một. Trả
list mới, không thay đổi list input tại chỗ. Một lượt merge có chi phí tuyến
tính theo chiều dài list.

### 9.4. `BPE.train(text, num_merges)`

1. `text.encode("utf-8")` chuyển sang byte; `list(...)` tạo IDs 0…255.
2. Khởi tạo `merges={}`, `vocab={i: bytes([i]) for i in range(256)}`.
3. Mỗi vòng đếm pair, dừng nếu không còn pair.
4. Chọn pair thường gặp nhất; dừng nếu tần suất dưới 2.
5. Tạo ID `256+k`, thay pair trong chuỗi train bằng `merge`.
6. Ghi `merges[pair]=new_id`; bytes của token mới là nối bytes của hai token cũ.
7. Trả `self` để có thể viết `BPE().train(...)`.

`num_merges` là số vòng tối đa; số token mới có thể ít hơn nếu hết pair đủ
thường gặp. Không cố định vocabulary bằng số từ. Không có lưu/load tokenizer
ra file trong CLI hiện tại.

### 9.5. `BPE.encode(text)`

Chuyển input mới sang byte IDs. Khi còn ít nhất hai token, tìm pair có luật
merge được học sớm nhất còn áp dụng được. Code dùng `min` theo ID của luật;
token mới được đánh số tăng dần nên ID nhỏ tương ứng merge sớm hơn. Pair
chưa biết có hạng vô cực; nếu không có luật áp dụng, dừng. Trả list IDs.

Phải gọi `train` trước `encode`. Khác tokenizer ký tự, byte vocabulary ban
đầu bao phủ các byte của UTF-8 nên không bỏ ký tự lạ theo điều kiện membership.

### 9.6. `BPE.decode(ids)`

Lookup mỗi ID thành bytes, nối tất cả rồi decode UTF-8 với `errors="replace"`.
Các IDs do encode hợp lệ tạo ra có thể ghép lại văn bản gốc. Khi decode từng
token riêng của một chữ tiếng Việt, một token có thể chỉ chứa một phần byte
sequence nên hiển thị ký tự thay thế; decode cả chuỗi mới là phép kiểm tra
round-trip chính. ID không có trong vocab sẽ gây lỗi lookup.

### 9.7. Khối chạy CLI cuối file

Parser đọc `--merges` mặc định 300 và `--text` tùy chọn. Nếu không có file,
corpus là các `LESSON.md` trong course. Script train BPE, in 15 merges đầu,
token dài, encode/decode một câu mẫu tiếng Anh và so với câu tiếng Việt.
Compression in ở đây là `số_byte_UTF8 / số_token`, không phải đánh giá chất
lượng model. Câu tiếng Việt có thể tách kém vì corpus chủ yếu tiếng Anh.

Vì mẫu round-trip trong CLI vẫn là câu tiếng Anh cố định, bạn nên tự thêm
kiểm tra trên chính corpus khi thực nghiệm ngôn ngữ khác. Không cần sửa BPE
để hoàn thành luồng GPT tokenizer ký tự của task hiện tại.

## 10. Toàn bộ `test_mini_gpt.py`

Test chạy offline, mỗi test xác nhận một thuộc tính kỹ thuật. “8 passed” cho
biết các thuộc tính này đạt, không phải model đạt 8/8 câu hỏi hay ngôn ngữ tốt.

### 10.1. Imports và cơ chế pytest

`pytest` tìm hàm `test_*`; các tham số `tmp_path`, `monkeypatch`, `capsys` là
fixture do pytest cấp tự động. `parametrize` chạy một hàm với nhiều bộ input.
Gọi `python -m pytest` dùng pytest trong đúng interpreter đã activate.
[Cách gọi pytest](https://docs.pytest.org/en/stable/how-to/usage.html) mô tả
chạy cả file, chọn một test và các tùy chọn output.

### 10.2. `test_character_tokenizer_round_trip(text)` — 2 test case

Chạy với `hello AI` và `xin chào AI`. Dựng vocabulary từ chính text, encode rồi
decode và assert bằng input. Điều này kiểm tra mapping/Unicode có trong vocab;
không chứng minh tokenizer xử lý được mọi ký tự ngoài corpus.

### 10.3. `test_batch_targets_are_next_tokens()` — 1 case

Tạo sequence `arange(100)`, lấy batch `[4,8]`. Vì sequence tăng đúng 1 ở mỗi
vị trí nên target phải bằng input+1. Assertion kiểm tra shift target của
`get_batch`; số trong ví dụ là dữ liệu giả để kiểm tra, không phải language corpus.

### 10.4. `test_future_tokens_do_not_change_prefix_logits()` — 1 case

Seed cố định, model hai block nhỏ ở eval. Hai input giống ba token đầu nhưng
khác hai token sau. Prefix logits phải bằng nhau trong tolerance; phần sau
phải có khác biệt để tránh một model không phản ứng với input vẫn pass dễ dàng.
Đây là kiểm tra hành vi causal xuyên qua toàn model, hữu ích hơn chỉ nhìn
mask có dạng tam giác. `no_grad` tiết kiệm graph cho test so sánh output.

### 10.5. `test_loss_backpropagates_through_embeddings_attention_and_ffn()` — 1 case

Tạo model nhỏ, input/target shift một token, gọi loss.backward. Kiểm tra gradient
của token embedding, position embedding, QKV và FFN up tồn tại, hữu hạn và có
độ lớn khác 0. Test xác nhận đường gradient tới các thành phần chính; chưa
bảo đảm mọi gradient đúng bằng một phép kiểm tra đạo hàm số, hoặc model đã hội tụ.

### 10.6. `test_cli_trains_and_samples_text_without_newline(...)` — 1 case

- `tmp_path`: nơi chứa corpus và checkpoint tạm, tách khỏi kết quả cá nhân.
- Corpus cố ý không có newline để kiểm tra fallback token đầu khi sinh.
- `monkeypatch.setattr(sys, "argv", ...)`: giả lập command-line một bước train.
- Lưu `original_generate` để vẫn chạy sampling thực tế.
- Hàm lồng `generate_in_eval_mode(model, *args, **kwargs)` assert model đang
  eval rồi gọi hàm generate gốc; đây là wrapper kiểm tra, không phải model khác.
- Gọi `main()` thật: tạo thư mục output mới, train, save và generate.
- Load checkpoint, dựng GPT từ config, nạp weights và forward kiểm tra số hữu hạn.
- `capsys` lấy stdout, tách mẫu sau `=== sample ===`; phải có 1 token khởi đầu
  cộng 4 token mới, tổng 5 ký tự vì đang dùng tokenizer ký tự.

`monkeypatch` tự khôi phục các thay đổi sau test. Checkpoint một bước này ở
thư mục tạm, khác `model_100.pt`/`model_500.pt` của thực nghiệm chính.

### 10.7. `test_cli_explains_invalid_setup(...)` — 2 case

Parameterization tạo hai cấu hình sai: embedding 30 không chia hết cho 4
head; hoặc corpus `hello` quá ngắn cho context 32 và split. `main()` phải thoát
với `SystemExit` code 2, stderr chứa thông báo có ý nghĩa. `pytest.raises`
coi lỗi dự kiến là kết quả đúng của test, không phải test bị crash.

Tổng: `2 + 1 + 1 + 1 + 1 + 2 = 8` case. Có thể chạy riêng causal test:

```bash
python -m pytest 02_llm_internals/test_mini_gpt.py::test_future_tokens_do_not_change_prefix_logits -v
```

## 11. Nạp model và sinh văn bản

Sau mục 5, chạy đoạn này để dùng checkpoint thật của máy mới. Trên PowerShell,
dùng file nháp theo mục 3.5. Đoạn này tự lưu output UTF-8, không cần pipe log.

```bash
python - <<'PY'
import sys
from pathlib import Path
import torch
sys.path.insert(0, "02_llm_internals")
from mini_gpt import Config, GPT

folder = Path("02_llm_internals/out/my_task01")
checkpoint = torch.load(folder / "model_500.pt", map_location="cpu", weights_only=True)
model = GPT(Config(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])
model.eval()
chars = checkpoint["chars"]
stoi = {char: i for i, char in enumerate(chars)}
prompt = "hello "
missing = sorted(set(prompt) - set(chars))
if missing:
    raise ValueError(f"Ký tự ngoài vocabulary: {missing}")
inputs = torch.tensor([[stoi[char] for char in prompt]], dtype=torch.long)
torch.manual_seed(42)
output = model.generate(inputs, max_new_tokens=120, temperature=0.8)
sample = "".join(chars[i] for i in output[0].tolist())
result = f"checkpoint=model_500.pt\nconfig={checkpoint['config']}\nprompt={prompt!r}\nseed=42 temperature=0.8 new_tokens=120\n{sample}\n"
(folder / "generate.log").write_text(result, encoding="utf-8")
print(result)
PY
```

Giải thích các bước nạp:

- `map_location="cpu"`: nạp tensor về CPU cho demo, kể cả checkpoint từ thiết bị khác.
- `weights_only=True`: checkpoint của lab chỉ cần tensors và cấu trúc dữ liệu đơn giản.
- `GPT(Config(...))`: tạo đúng số layer/kích thước trước khi nạp trọng số.
- `load_state_dict`: thay weights khởi tạo bằng weights đã học và nạp buffer.
- Dùng `chars` từ cùng checkpoint để IDs có đúng nghĩa; không dựng vocabulary khác.
- `eval`: tắt dropout, rồi `generate` thực hiện forward/sampling không gradient.

Thay đổi prompt hoặc sampling seed/temperature có thể cho mẫu khác dù model
giữ nguyên. Đó là thay đổi inference, không phải train lại. Tokenizer hiện
không có ký tự ngoài vocab; kiểm tra `missing` bảo đảm prompt không bị mất chữ.

Sau train, bạn có **một model ngôn ngữ nhỏ đã học trọng số**. File `.pt` chứa
weights/config/chars; cần implementation GPT tương ứng để dùng. Nó chưa phải
một chatbot instruction-tuned, chưa có server/API hoặc giao diện phục vụ.

## 12. Đọc kết quả và làm thực nghiệm

### 12.1. Đọc một dòng log

```text
step   100 | train 2.306 | val 2.347 | lr 3.0e-04 | 1s
```

Ở bước 100, loss đo trên batch mẫu train/val lần lượt 2.306 và 2.347; learning
rate hiệu dụng là 0,0003; training loop đã chạy khoảng 1 giây theo phép đo này.
Mốc này không bao gồm toàn bộ thời gian cài package, import hoặc sinh văn bản.

### 12.2. Kết quả cũ chỉ là mốc đối chiếu

| Ngân sách | Tham số | Train loss đầu/cuối | Val loss đầu/cuối | Tổng thời gian lệnh máy cũ |
|---|---:|---|---|---|
| 100 bước | 104.384 | 3.593 → 2.306 | 3.591 → 2.347 | Khoảng 2,5 giây |
| 500 bước | 104.384 | 3.593 → 1.532 | 3.591 → 2.373 | Khoảng 4,7 giây |

Đây là các lần kiểm chứng trước trên CPU, corpus 2.004 ký tự, không phải cam
kết thời gian/kết quả trên máy mới. Cài dependency hoặc tải dữ liệu cần thời
gian riêng. Model lớn hơn/context dài hơn/nhiều bước hơn sẽ có chi phí khác.

Trong lần 500 bước cũ, val thấp nhất ở các mốc được đo là 2.318 tại bước 100,
sau đó dao động trong khi train loss tiếp tục giảm. Dấu hiệu này phù hợp
overfitting trên corpus nhỏ; chưa đủ kết luận về mọi dữ liệu hoặc mọi seed.
Script không giữ checkpoint tại mốc val tốt nhất, chỉ giữ bước cuối.

### 12.3. Tại sao loss giảm mà câu vẫn lộn xộn?

Loss đo dự đoán token kế tiếp, không trực tiếp chấm câu đúng ngữ nghĩa. Model
có thể học tần suất ký tự, cụm thường gặp và dấu cách trước khi học được câu
mạch lạc. Corpus nhỏ, context 32 ký tự và model nhỏ giới hạn việc học. Nhận xét
mẫu sinh cần đi cùng loss, không thay bằng kết luận “AI đã hiểu toàn bộ corpus”.

### 12.4. Tự code theo từng phần

Giữ lịch sử Git trước khi thực hành. Trong chính file `mini_gpt.py`, chọn một
hàm, giải thích input/output, tự viết lại từ cách hiểu rồi chạy test liên quan.
Thứ tự hợp lý: tokenizer → get_batch → MLP → Block → attention → GPT.forward
→ train/generate. Mỗi lần sửa một phần để biết thay đổi nào tạo ra kết quả.
Không cần thêm nhiều bản model trùng nhau vào repo để làm task này.

Thực nghiệm gợi ý sau khi có baseline:

| Thay đổi | Cần quan sát | Phần kiến thức |
|---|---|---|
| Giảm/tăng lr, giữ cấu hình khác | Tốc độ giảm loss, dao động | Optimization |
| Đổi n_layer 2 → 1 | Số tham số, train/val | Capacity |
| Đổi n_head 4 → 2, giữ C=64 | Param count giữ nguyên, kết quả có thể đổi | Chia head |
| Đổi context 32 → 16 | Chi phí attention và khả năng dùng ngữ cảnh | Context |
| Bỏ causal mask trong nhánh thực nghiệm | Causal test phải fail | Leakage tương lai |
| Tăng ngân sách train | Khoảng cách train/val | Overfitting |

Thí nghiệm bỏ mask dùng để hiểu/test, không dùng checkpoint đó làm kết quả
decoder-only hợp lệ. Chọn tên log/checkpoint khác cho các biến thể. Không đo
so sánh dataset khác nhau bằng loss như thể mọi token/vocabulary giống nhau.

### 12.5. Dùng Tiny Shakespeare hoặc corpus tự chuẩn bị

Sau baseline, có thể đổi `--data` sang `02_llm_internals/data/my_corpus.txt`,
file plain text UTF-8 đủ dài. Script không đọc PDF/Word/JSON thành ngôn ngữ tự
nhiên tự động; nếu truyền JSON, nó sẽ học cả cú pháp JSON như văn bản.

Tiny Shakespeare cache không theo Git. Để tải rõ ràng và dùng trong lệnh riêng:

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
destination = Path("02_llm_internals/data/tinyshakespeare.txt")
destination.parent.mkdir(parents=True, exist_ok=True)
urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", destination)
print(destination, destination.stat().st_size)
PY
```

Lệnh cần mạng. Đổi `--data` trong lệnh train, đồng thời đổi tên output và log;
ghi lại hash/version corpus. Với cấu hình mẫu cố định, mỗi step vẫn xử lý
B*T vị trí; thêm dữ liệu không tự tăng số step. Thực nghiệm mới cần được
đánh giá riêng thay vì chép số liệu của `sample_corpus.txt` vào báo cáo đó.

## 13. Tạo báo cáo của máy mới

### 13.1. Thu thập kết quả từ log và checkpoint thực tế

Đoạn dưới chỉ tổng hợp, **không train thêm**. Chạy sau mục 5 và 11. Nó kiểm
tra log đã tới đúng bước, checkpoint nạp được và output hữu hạn; ghi metadata,
hash source/corpus, số tham số và báo cáo `.md` bằng số liệu máy mới.

Trên PowerShell dùng file nháp. Ghi nhận môi trường và `packages.txt` phải
có sẵn từ mục 5.1. Nếu cố ý đổi cấu hình, cập nhật kỳ vọng baseline/ghi chú
trong script trước khi dùng; đoạn này kiểm tra hai lần chạy mốc 100/500.

```bash
python - <<'PY'
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch
sys.path.insert(0, "02_llm_internals")
from mini_gpt import Config, GPT

root = Path("02_llm_internals")
folder = root / "out/my_task01"

def read_log(path):
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")

def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
corpus = root / "sample_corpus.txt"
text = corpus.read_text(encoding="utf-8")
digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
if digest != environment["dataset_sha256"]:
    raise ValueError("Corpus thay đổi từ lúc ghi môi trường; kiểm tra và chạy lại thực nghiệm.")
tests = read_log(folder / "tests.log")
if not re.search(r"\b8 passed\b", tests) or re.search(r"\d+ failed|ERROR", tests):
    raise ValueError("Cần log 8 test passed trước khi tổng hợp baseline.")

runs = []
for steps in (100, 500):
    output = read_log(folder / f"train_{steps}.log")
    if "Traceback" in output or "=== sample ===" not in output:
        raise ValueError(f"Lần {steps} chưa hoàn tất hoặc có lỗi.")
    pattern = r"step\s+(\d+) \| train ([0-9.]+) \| val ([0-9.]+) \| lr ([0-9.eE+-]+) \| (\d+)s"
    history = [{"step": int(s), "train": float(t), "validation": float(v),
                "lr": float(lr), "loop_seconds": int(seconds)}
               for s, t, v, lr, seconds in re.findall(pattern, output)]
    if not history or history[0]["step"] != 0 or history[-1]["step"] != steps:
        raise ValueError(f"Log {steps} không đủ bước 0..{steps}.")
    checkpoint_path = folder / f"model_{steps}.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = Config(**saved["config"])
    if not (cfg.n_layer == 2 and cfg.n_head == 4 and cfg.n_embd == 64 and cfg.block_size == 32 and cfg.dropout == 0.1):
        raise ValueError("Checkpoint khác cấu hình baseline; báo cáo thành thực nghiệm riêng.")
    if saved["chars"] != sorted(set(text)) or cfg.vocab_size != len(saved["chars"]):
        raise ValueError("Vocabulary không khớp corpus.")
    model = GPT(cfg).eval()
    model.load_state_dict(saved["model"])
    stoi = {c: i for i, c in enumerate(saved["chars"])}
    ids = [stoi[c] for c in text[:cfg.block_size + 1]]
    with torch.no_grad():
        logits, loss = model(torch.tensor([ids[:-1]]), torch.tensor([ids[1:]]))
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise ValueError("Checkpoint có output/loss không hữu hạn.")
    runs.append({"steps": steps, "config": saved["config"],
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                 "checkpoint_reload": "passed", "probe_shape": list(logits.shape),
                 "history": history, "sample": output.split("=== sample ===", 1)[1].strip()})

sources = ["mini_gpt.py", "bpe.py", "test_mini_gpt.py"]
metadata = {"recorded_at": datetime.now().astimezone().isoformat(),
            "environment": environment, "dataset_sha256": digest,
            "characters": len(text), "vocabulary": len(set(text)),
            "git_status_at_summary": git_value("status", "--short"),
            "source_sha256_at_summary": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in sources},
            "runs": runs}
(folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Báo cáo thực nghiệm Task 01 — Máy mới", "",
         "Người thực hiện: [điền tên]. Ngày chạy: " + metadata["recorded_at"], "",
         "## 1. Mục tiêu và phần tự triển khai", "",
         "[Mô tả tokenizer, decoder-only, các hàm bạn tự viết/sửa; nêu phần dựa trên lab.]", "",
         "## 2. Môi trường, dữ liệu và kiến trúc", "",
         f"Python: {environment['python']}. PyTorch: {environment['torch']}.", "",
         f"Corpus: {len(text)} ký tự, vocabulary {len(set(text))}; split 90/10.", "",
         "2 block, 4 head, embedding 64, context 32, dropout 0.1; CPU; batch 8; lr cơ sở 0.003; seed 1337.", "",
         "[Giải thích attention, position embedding, FFN, residual và kích thước tensor.]", "",
         "## 3. Kết quả đo thực tế", "",
         "| Steps | Tham số | Train đầu/cuối | Val đầu/cuối | Giây trong vòng train cuối | Reload |",
         "|---|---:|---|---|---:|---|"]
for run in runs:
    first, last = run["history"][0], run["history"][-1]
    lines.append(f"| {run['steps']} | {run['parameters']} | {first['train']} → {last['train']} | {first['validation']} → {last['validation']} | {last['loop_seconds']} | passed |")
lines += ["", f"Mốc đoán đều ln(V) = {math.log(len(set(text))):.3f}.", "",
          "Số giây trong bảng do training loop in, làm tròn; không phải tổng thời gian setup/chạy lệnh.", ""]
for run in runs:
    lines += [f"### Mẫu sinh sau {run['steps']} bước", "", "```text", run["sample"], "```", ""]
lines += ["## 4. Kiểm thử và bằng chứng", "",
          "- [Test log](tests.log), [môi trường](environment.json), [packages](packages.txt), [metadata](metadata.json).",
          "- [Train 100](train_100.log), [train 500](train_500.log), [inference](generate.log).", "",
          "8 test passed; hai checkpoint reload thành công. Giải thích ý nghĩa và giới hạn của các test.", "",
          "## 5. Phân tích, hạn chế và cải tiến", "",
          "[Đối chiếu train/val, nhận xét văn bản, corpus nhỏ, overfitting, giới hạn context và tokenizer.]", "",
          "## 6. Cách tái hiện", "",
          "Các lệnh baseline tại [hướng dẫn đầy đủ](../../TASK_01_FULL_GUIDE.md), mục 5 và 11.", "",
          "Lưu source/config/metadata cùng bài nộp. Lần 500 bước khởi tạo mới; schedule phụ thuộc steps.", ""]
report = folder / "REPORT.md"
if report.exists():
    report = folder / "REPORT_generated.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("Đã ghi:", report)
print("Metadata:", folder / "metadata.json")
PY
```

Lần đầu tạo `REPORT.md`. Các lần sau tạo/ghi `REPORT_generated.md` để giữ
phần phân tích bạn đã viết trong `REPORT.md`; bạn tự đối chiếu và cập nhật
bản sẽ nộp. Các log/metadata chỉ là bằng chứng của source thực sự dùng lúc
chạy. Hash source ở đoạn này được đo tại lúc tổng hợp: chạy tổng hợp ngay
sau thực nghiệm, không sửa source giữa train và tổng hợp.

### 13.2. Bạn phải bổ sung gì vào báo cáo tự sinh?

- Tên, ngày thực hiện, mục tiêu và phần tự cài đặt/thay đổi từ code lab.
- Sơ đồ kiến trúc và ý nghĩa B/T/C/H/D/V; cách shift target và che tương lai.
- Cấu hình đầy đủ, loại dữ liệu, split, source commit và phiên bản thư viện.
- Loss đầu/cuối, nhận xét mẫu sinh, bằng chứng test/reload từ máy mới.
- Phân tích train/val, điều gì chứng minh được và điều gì chưa chứng minh.
- Hạn chế và một cải tiến có thể thực hiện, ví dụ best checkpoint/early stopping.

Mẫu trong `TASK_01_REPORT.md` ở source là tài liệu tham khảo. Bản nộp của
máy mới nằm ở `out/my_task01/REPORT.md`; không cần giữ các đường dẫn output
cũ trong báo cáo mới. Chỉ đánh dấu checklist năng lực khi bạn tự giải thích được.

## 14. Đóng gói bài nộp

Đoạn Python dưới dùng `zipfile`, chạy được trên ba OS theo cách nhập lệnh
mục 3.5. Nó chỉ chọn source cần cho lab và thư mục thực nghiệm mới; không
phụ thuộc `out/task01_review/` hoặc checkpoint của máy cũ.

Chạy sau khi đã hoàn thiện `REPORT.md`. Nó tạo mới ZIP mỗi lần, kiểm tra CRC
và có các assert giúp nhận biết file cần thiết còn thiếu:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path("02_llm_internals")
folder = root / "out/my_task01"
required = ["REPORT.md", "environment.json", "packages.txt", "metadata.json",
            "tokenizer.log", "bpe.log", "tests.log", "train_100.log", "train_500.log",
            "generate.log", "model_100.pt", "model_500.pt"]
for name in required:
    if not (folder / name).is_file():
        raise FileNotFoundError(f"Thiếu bằng chứng: {folder / name}")
sources = ["TASK_01_FULL_GUIDE.md", "TASK_01_REVIEW.ipynb", "TASK_01.md", "LESSON.md", "mini_gpt.py",
           "bpe.py", "test_mini_gpt.py", "sample_corpus.txt"]
archive = folder / "submission.zip"
with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
    bundle.writestr("README.md", "# Task 01 review\n\nBáo cáo: 02_llm_internals/out/my_task01/REPORT.md\n\nHướng dẫn chạy: 02_llm_internals/TASK_01_FULL_GUIDE.md\n\nSau giải nén, chạy từ thư mục chứa 02_llm_internals/.\n")
    for name in sources:
        path = root / name
        bundle.write(path, path.as_posix())
    for name in required:
        path = folder / name
        bundle.write(path, path.as_posix())
with ZipFile(archive) as bundle:
    if bundle.testzip() is not None:
        raise ValueError("Archive lỗi CRC")
    print("Số file:", len(bundle.namelist()))
print("Bài nộp:", archive.resolve())
print("Dung lượng byte:", archive.stat().st_size)
PY
```

ZIP chứa cả checkpoint vì đó là artifact bài nộp; `.pt` và `out/` vẫn không
được commit Git. Nếu muốn lưu bản báo cáo kết quả vào repo, copy bản đã hoàn
thiện sang một file Markdown trong thư mục được track và dùng đường dẫn
artifact phù hợp nơi nộp; không force-add môi trường/cache/checkpoint.

Khi clone repo trên máy reviewer, làm lại setup theo mục 3. Khi reviewer giải
nén ZIP, chạy từ thư mục có `02_llm_internals/`; đây là cùng layout tương đối.
Sau khi cập nhật báo cáo hoặc chạy lại model, chạy lại bước tổng hợp/đóng gói.

## 15. Lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Clone xong không có `.pt`/log | Git ignore artifact | Chạy lại mục 5; không cần artifact cũ |
| `cd course` không tồn tại sau clone | Đã ở root của chính repo course | Kiểm tra thư mục có `02_llm_internals/` |
| `python`/`py` không được nhận | Python chưa cài/PATH chưa đúng | Kiểm tra bộ cài, mở terminal mới, xác nhận version |
| `No module named torch` hoặc `pytest` | Cài vào interpreter khác | Activate, xem `sys.executable`, cài bằng `python -m pip` |
| `No matching distribution found` | Wheel không phù hợp Python/OS/kiến trúc/index | Dùng Python hỗ trợ, venv mới và bộ chọn PyTorch chính thức |
| `Activate.ps1 cannot be loaded` | Chính sách PowerShell | Dùng `.venv\Scripts\python.exe` trực tiếp |
| `No module named venv`/`ensurepip` | Bản Python thiếu thành phần venv | Cài gói venv tương ứng trên OS, tạo lại môi trường |
| PowerShell báo lỗi ở `<<` hoặc `\` | Copy cú pháp Bash sang PowerShell | Dùng lệnh một dòng hoặc file nháp Python |
| `No such file` | Working directory sai hoặc chưa tạo output | Kiểm tra root repo và mục 5.1 |
| Split quá ngắn | Corpus không đủ cho context | Dùng corpus mẫu hoặc thêm dữ liệu/giảm context |
| `n_embd must be divisible...` | Số chiều chia head không nguyên | Dùng 64/4 hoặc một cặp hợp lệ |
| CUDA/MPS unavailable | Device/wheel/driver không khả dụng | Quay về CPU để hoàn thành baseline |
| Shape mismatch khi load | Config, vocabulary hoặc source khác checkpoint | Dùng config/chars cùng checkpoint và implementation tương ứng |
| `KeyError` của prompt | Có ký tự ngoài vocab | Kiểm tra `missing`, dùng prompt được hỗ trợ |
| NaN/Inf | LR/config/dữ liệu có vấn đề | Quay về baseline, xem logits/loss và thử lr nhỏ hơn |
| UTF-8 decode lỗi khi đọc log | PowerShell cũ ghi UTF-16 | Dùng hàm `read_log` có phát hiện BOM ở mục 13 |
| Chạy lại không khớp số thập phân | OS, version, RNG, device hoặc config đổi | Ghi metadata, giữ seed và log interval, so xu hướng và nguyên nhân |
| Text sinh chưa thành câu | Model/corpus nhỏ, objective next-token | Báo cáo trung thực mẫu sinh và giới hạn |

Nếu bị lỗi mới, giữ traceback đầy đủ và command đã dùng; không chỉ giữ dòng
cuối. Đổi config trước khi ghi nhận bằng chứng sẽ làm khó tái hiện vấn đề.

## 16. Câu hỏi review và checklist

### 16.1. Các câu hỏi bạn cần tự trả lời

| Câu hỏi | Ý chính cần giải thích |
|---|---|
| Đầu vào là file test Python hay văn bản? | Corpus `.txt` qua `--data`; file test kiểm tra implementation |
| Token trong model này là gì? | Ký tự/code point có trong vocabulary, không mặc định là từ |
| Dùng thư viện tokenizer nào? | Tự viết CharTokenizer; BPE là lab tự viết riêng |
| Vì sao target lệch một token? | Học next-token ở từng vị trí |
| Vì sao causal mask cần thiết? | Che đáp án tương lai khi train song song |
| Q/K/V là gì? | Ba projection học được, scores từ QKᵀ, weights trộn V |
| Vì sao chia sqrt(D)? | Điều chỉnh độ lớn dot-product theo chiều head |
| Embedding 64 mà head ra 35? | 64 là chiều biểu diễn, 35 là số lựa chọn token |
| Positional encoding cụ thể là loại gì? | Learned position embedding, cộng vào token embedding |
| FFN/attention khác nhau? | FFN xử lý mỗi vị trí; attention trao đổi giữa các token |
| Residual và LayerNorm có vai trò gì? | Giữ luồng thông tin/gradient; chuẩn hóa đặc trưng |
| Các block là encoder hay decoder-only? | Causal self-attention, không có encoder/cross-attention |
| Ai cập nhật weights? | AdamW.step sau khi backward tính gradient |
| Train 500 bước có tăng số tham số? | Không; cùng kiến trúc/vocab vẫn 104.384 |
| `eval()` có giống `no_grad()`? | Không: đổi hành vi layer so với tắt graph |
| Vì sao train loss thấp chưa đủ? | Có thể overfit; cần val, mẫu sinh và đánh giá phù hợp |
| Checkpoint gồm gì? | state_dict, config, chars; chưa có đầy đủ state để resume chính xác |
| `8 passed` chứng minh gì? | Các thuộc tính kỹ thuật được test, không phải chất lượng ngôn ngữ |
| 500 bước có tiếp tục từ lần 100? | CLI hiện khởi tạo mới mỗi lần |
| Dùng được như chatbot chưa? | Đây là next-character model demo; chưa instruction-tuned hoặc serving |

### 16.2. Demo review 5–10 phút

1. Mở báo cáo máy mới, nêu mục tiêu và phần bạn tự cài đặt.
2. Mở source, đi từ tokenizer → embedding/vị trí → block → head → loss.
3. Chỉ ra causal mask và giải thích test future-token invariance.
4. Trình bày config và cách tính 104.384 tham số.
5. Mở log 100/500, so train/val, nhận xét overfitting và giới hạn thí nghiệm.
6. Nạp checkpoint, sinh từ prompt để chứng minh inference không cần train lại.
7. Nêu hạn chế và cải tiến tiếp theo; cung cấp ZIP để người review tái hiện.

### 16.3. Checklist trước khi nộp

- [ ] Hướng dẫn và source đã commit/push; máy mới clone được.
- [ ] Tạo `.venv` riêng trên máy mới, ghi đúng interpreter/version/device.
- [ ] Corpus rõ nguồn, hash rõ; không nhầm file Python test là dataset.
- [ ] Có 8 test passed trên implementation sẽ nộp.
- [ ] Có hai log hoàn tất, hai checkpoint và demo reload trên máy mới.
- [ ] Metadata/báo cáo lấy số liệu thực tế, không chép kết quả cũ thành của mình.
- [ ] Trình bày được từng hàm và shape thay vì chỉ chạy lệnh.
- [ ] Nêu rõ phần tự viết/sửa và phần dựa trên lab; giữ source/commit tương ứng.
- [ ] `REPORT.md` đã thay các phần `[điền...]` và có phân tích của người thực hiện.
- [ ] ZIP kiểm tra thành công và chứa log/checkpoint/source mới nhất.

## 17. Tài liệu chính thức

Ưu tiên source trong repo để hiểu chính xác bài này. Tài liệu dưới giúp đối
chiếu thuật ngữ/API; trang stable và lệnh cài đặt có thể thay đổi theo thời gian.

- [Python venv](https://docs.python.org/3.11/library/venv.html): môi trường riêng và activation.
- [PyTorch install selector](https://pytorch.org/get-started/locally/): chọn wheel theo OS/device.
- [PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/): tham khảo index CPU/CUDA.
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): logits và target.
- [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html): gradient mode và eval.
- [pytest invocation](https://docs.pytest.org/en/stable/how-to/usage.html): chạy test bằng interpreter.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): nền tảng attention/Transformer;
  lab hiện dùng decoder-only, Pre-LayerNorm và learned position embedding.

Sau khi hoàn thành bài, có thể học tiếp dataset engineering/fine-tuning/evaluation
trong các module còn lại. Task hiện tại tập trung vào kiến trúc và cơ chế học
của một mô hình ngôn ngữ nhỏ.
