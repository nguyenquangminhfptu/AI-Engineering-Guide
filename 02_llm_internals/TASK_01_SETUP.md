# Task 01 — Setup và chạy trên máy mới

Dùng file này để bắt đầu. Chạy và trình bày bằng
[TASK_01_REVIEW.ipynb](TASK_01_REVIEW.ipynb); khi cần hiểu từng class/function,
tra cứu [TASK_01_FULL_GUIDE.md](TASK_01_FULL_GUIDE.md).

**Bạn chỉ mở/chạy notebook, nhưng cần giữ đủ các file phụ thuộc.** Cách dễ
nhất là clone cả repo. Nếu copy thủ công, giữ nguyên cấu trúc tối thiểu sau:

```text
thu-muc-repo/
└── 02_llm_internals/
    ├── TASK_01_REVIEW.ipynb  # File bạn mở và chạy
    ├── sample_corpus.txt    # Dữ liệu train
    ├── mini_gpt.py          # Model và hàm train được notebook import
    ├── test_mini_gpt.py     # Các test notebook sẽ gọi
    └── bpe.py               # Notebook đọc hash khi tạo metadata
```

Chỉ copy `.txt` và `.ipynb` thì chưa đủ cho notebook hiện tại. Các file Python
đi cùng được dùng tự động; không cần chạy riêng trước khi mở notebook.
Mang thêm file setup này và hướng dẫn đầy đủ để tra cứu.

## 1. Chuẩn bị Git và Python

Cài **Git** và **Python 3.11** nếu máy chưa có. Mở Terminal trong IDE.

Kiểm tra Git:

```bash
git --version
```

Kiểm tra Python bằng `python3 --version` trên macOS/Linux hoặc
`py -3.11 --version` trên Windows. Task chạy bằng CPU, không cần API key.

## 2. Clone repo

```bash
git clone https://github.com/nguyenquangminhfptu/AI-Engineering-Guide.git
cd AI-Engineering-Guide
```

Đây là root repo, nơi có `README.md` và `02_llm_internals/`. Giữ terminal tại
đây trong các bước sau; không chạy thêm `cd course`.

Nếu đã clone trước đó, mở terminal tại root repo và cập nhật bằng:

```bash
git pull --ff-only
```

## 3. Tạo và kích hoạt môi trường

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Nếu `python3` không phải Python 3.11, dùng `python3.11 -m venv .venv` sau khi
cài phiên bản đó.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn activation, dùng `.\.venv\Scripts\python.exe` thay cho
`python` trong các lệnh tiếp theo. Các bước pip/kernel/Jupyter bên dưới đều
dùng được với đường dẫn interpreter trực tiếp này.

## 4. Cài thư viện

Trước tiên:

```bash
python -m pip install --upgrade pip
```

**macOS:**

```bash
python -m pip install torch pytest jupyterlab matplotlib
```

**Windows/Linux — PyTorch CPU:**

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest jupyterlab matplotlib
```

- `torch`: model, tensor, gradient và optimizer.
- `pytest`: kiểm tra implementation.
- `jupyterlab`: mở/chạy notebook; cài kèm kernel Python cần dùng.
- `matplotlib`: vẽ biểu đồ loss.

Chỉ cần các package trên cho bài này; `requirements.txt` ở root dành cho cả
course, có thêm dependencies của các module khác.

Kiểm tra sau khi cài thành công:

```bash
python -c "import sys, torch, pytest, jupyterlab, matplotlib; print(sys.executable); print('torch:', torch.__version__); print('Dependencies: OK')"
```

Đường dẫn Python phải thuộc `.venv/bin/` hoặc `.venv\Scripts\` của repo.
Nếu tải package bị timeout, thử lại lệnh pip đó với tùy chọn `--timeout 120`.

## 5. Đăng ký kernel và mở notebook

```bash
python -m ipykernel install --user --name task01 --display-name "Task 01 (.venv)"
python -m jupyterlab 02_llm_internals/TASK_01_REVIEW.ipynb
```

JupyterLab mở trong trình duyệt. Chọn kernel **Task 01 (.venv)** trong notebook.
Giữ terminal chạy JupyterLab trong lúc làm bài.

## 6. Điền thông tin và chạy toàn bài

Trong cell cấu hình, điền tên và giữ CPU cho baseline:

```python
AUTHOR = "Tên của bạn"
DEVICE = "cpu"
```

Giữ `RUN_NAME = "notebook_task01"` và corpus mặc định `sample_corpus.txt` trong
lần chạy đầu. File corpus đã có trên Git; chưa cần `data/tinyshakespeare.txt`.

Chọn **Restart Kernel and Run All**. Notebook tự thực hiện theo thứ tự:

1. Import thư viện, tìm root repo và tạo thư mục output.
2. Đọc corpus, dựng tokenizer ký tự và chia train/validation.
3. Khởi tạo GPT, hiển thị shape tensor và số tham số.
4. Kiểm tra causal mask và chạy pytest.
5. Train hai model mới với ngân sách 100/500 bước.
6. Vẽ loss, so sánh kết quả và lưu checkpoint.
7. Nạp checkpoint từ đĩa, sinh văn bản và lưu metadata/báo cáo.

`mini_gpt.py`, `test_mini_gpt.py`, `sample_corpus.txt` và `bpe.py` phải ở cùng
thư mục module như khi clone. Notebook import model, gọi test và đọc hash các
file source; bạn không cần chạy các script riêng trước.

Kết quả mốc: model 104.384 tham số và test gốc có 8 case passed. Số liệu loss
và thời gian cần lấy từ lần chạy máy mới. Các output đã lưu sẵn trong notebook
là kết quả kiểm chứng trước, không thay thế việc chạy lại trên máy báo cáo.

## 7. Lưu kết quả và hoàn thiện báo cáo

Sau khi mọi cell hoàn tất, nhấn **Ctrl+S** hoặc **Cmd+S** để lưu notebook có
output. Các file mới được tạo tại:

```text
02_llm_internals/out/notebook_task01/
├── model_100.pt
├── model_500.pt
├── train_100.log
├── train_500.log
├── generate_100.log
├── generate_500.log
├── tests.log
├── loss_curves.png
├── metadata.json
└── REPORT.md
```

Điền phần tự triển khai và nhận xét của bạn vào `REPORT.md` và phần cuối
notebook. Nếu chạy lại khi đã có `REPORT.md`, notebook ghi bản nháp mới vào
`REPORT_generated.md` để giữ bản bạn đã biên tập; đối chiếu rồi cập nhật bản nộp.

Thứ tự trình bày review: **dữ liệu → tokenizer → kiến trúc/shape → causal mask
và test → train/val loss → demo reload → hạn chế và cải tiến**.

Source, notebook và hướng dẫn được Git lưu khi commit/push. `.venv/`, `data/`,
`out/` và checkpoint `.pt` được Git bỏ qua; máy mới tự cài/tạo lại. Đính kèm
log, biểu đồ, báo cáo và checkpoint như artifacts của bài nộp.

Khi cần giải thích sâu hoặc xử lý lỗi, xem [hướng dẫn đầy đủ](TASK_01_FULL_GUIDE.md).
