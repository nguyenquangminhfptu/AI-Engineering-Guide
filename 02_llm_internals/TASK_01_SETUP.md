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

## 8. Thử khả năng model đã train

Dùng [probe_model.py](probe_model.py) để nạp checkpoint, nhập prompt và đo
next-character prediction mà không train lại. Từ root repo:

```bash
python 02_llm_internals/probe_model.py --checkpoint 02_llm_internals/out/notebook_task01/model_500.pt
```

Gõ `hello `, `AI is ` hoặc `the model ` tại `Prompt>`. Model sẽ in 5 ký tự có
xác suất cao nhất và viết tiếp văn bản. Gõ `/quit` để thoát. Nếu bạn dùng bản
chạy thử cá nhân, thay `out/notebook_task01/` bằng `out/local_trial_20261026/`.

Thử một prompt cố định với tham số sampling:

```bash
python 02_llm_internals/probe_model.py \
  --checkpoint 02_llm_internals/out/notebook_task01/model_500.pt \
  --prompt "hello " --new-tokens 80 --temperature 0.8 --seed 42
```

Trên PowerShell, ghép các dòng thành một dòng, bỏ ký tự nối dòng `\`.
Có thể đổi temperature thành 0.3 hoặc 1.2 để quan sát phân phối sinh; giá trị
phải lớn hơn 0. Đổi sampling không cập nhật trọng số. Prompt phải chứa các
ký tự có trong vocabulary; công cụ báo lỗi thay vì tự bỏ ký tự chưa biết.

Đo trên 10% cuối corpus gốc, cùng split với lúc train:

```bash
python 02_llm_internals/probe_model.py \
  --checkpoint 02_llm_internals/out/notebook_task01/model_500.pt \
  --eval-text 02_llm_internals/sample_corpus.txt --split val \
  --out 02_llm_internals/out/notebook_task01/probe_500.json
```

Đổi checkpoint và tên JSON sang `model_100.pt`/`probe_100.json` để so sánh.
Các chỉ số là cross-entropy, perplexity, top-1 và top-5 accuracy của **ký tự
tiếp theo**, không phải độ đúng của câu trả lời. Evaluation chấm từng mục
tiêu một lần với tối đa context 32; khác cách ước lượng batch ngẫu nhiên trong
log train nên loss có thể khác. Perplexity thấp hơn tốt hơn trên cùng dữ liệu
và tokenizer; accuracy cao hơn tốt hơn theo tiêu chí chọn đúng ký tự cao nhất.

Với file văn bản mới chưa dùng train, dùng `--split all` để đánh giá toàn file.
Dataset phải đủ dài và có các ký tự trong vocabulary. Corpus/validation hiện
tại rất nhỏ, chưa đủ đánh giá tổng quát hoặc khả năng chatbot.

## 9. Thực nghiệm nhiều dữ liệu hơn với Tiny Shakespeare

Corpus mẫu 2.004 ký tự phù hợp kiểm tra luồng train. Muốn quan sát model học
chính tả và các cụm dài hơn, làm một thực nghiệm riêng với Tiny Shakespeare
(bản đang dùng có 1.115.394 ký tự và 65 ký tự khác nhau). Văn bản đích là tiếng
Anh theo phong cách kịch, không phải hội thoại hỏi đáp hoặc tiếng Việt.

Nếu máy mới chưa có cache, tải corpus bằng lệnh một dòng này từ root repo:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; p=Path('02_llm_internals/data/tinyshakespeare.txt'); p.parent.mkdir(parents=True, exist_ok=True); urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', p); print(p)"
```

Chạy thử cấu hình lớn hơn, lưu model ở thư mục khác để giữ baseline:

```bash
python -u 02_llm_internals/mini_gpt.py \
  --data 02_llm_internals/data/tinyshakespeare.txt --device cpu \
  --n_layer 4 --n_head 4 --n_embd 128 --block_size 128 \
  --batch_size 8 --steps 3000 --lr 0.001 --dropout 0.1 \
  --log_every 200 --sample_tokens 220 \
  --out 02_llm_internals/out/shakespeare_v1/model.pt
```

Trên PowerShell ghép thành một dòng, bỏ các ký tự nối dòng `\`. Cấu hình này
có 818.048 tham số khi vocabulary là 65; tốn nhiều thời gian hơn baseline.
Thời gian phụ thuộc CPU và tải máy, không dùng thời gian vài giây của model
104.384 tham số để ước lượng trực tiếp.

Lần kiểm chứng ngày 26/10/2026 với cấu hình này đạt train loss 1.777 và val
loss 1.915 tại bước 3000, từ các mốc ban đầu 4.229/4.226. Log ghi tổng thời
gian vòng train 1179 giây (~19,7 phút), gồm một khoảng tăng thời gian lớn chưa
rõ nguyên nhân. Mẫu sinh có thêm từ/cấu trúc tiếng Anh nhưng vẫn sai chính tả
và ngữ pháp; đây chưa phải cấu hình bảo đảm tạo câu hoàn chỉnh có nghĩa.

Thử checkpoint mới sau khi lệnh train hoàn tất:

```bash
python 02_llm_internals/probe_model.py \
  --checkpoint 02_llm_internals/out/shakespeare_v1/model.pt \
  --prompt "the " --new-tokens 160 --temperature 0.7 --top-k 10
```

Dữ liệu nhiều hơn cần đi cùng đủ ngân sách học, context/capacity phù hợp và
theo dõi validation. Tăng số bước trên corpus cũ có thể làm tăng overfitting.
Thực nghiệm trên thay cả dữ liệu lẫn cấu hình, nên dùng để quan sát kết quả
của cấu hình mới, không quy toàn bộ cải thiện cho riêng lượng dữ liệu.

Loss trên corpus mới có vocabulary khác không được so trực tiếp như hai lần
đánh giá cùng dữ liệu/tokenizer. Giữ log, cấu hình và mẫu sinh của từng thực
nghiệm để giải thích khi review. Model nhỏ vẫn có thể sinh sai từ/ngữ pháp.

Ngoài chất lượng ngôn ngữ, `generate` hiện dừng theo số ký tự mới, nên có thể
cắt giữa từ/câu. Muốn định dạng đầu ra thành trọn câu cần cơ chế dừng theo dấu
kết câu hoặc EOS phù hợp. Cắt ở dấu chấm chỉ sửa cách dừng, không chứng minh
câu đúng ngữ pháp/ngữ nghĩa. Corpus tốt, train tốt và quy tắc dừng là các phần
riêng cần xem xét.
