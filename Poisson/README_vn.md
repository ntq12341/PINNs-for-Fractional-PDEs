# Poisson fPINN

Thư mục này dùng để tái lập các thí nghiệm Poisson 1D trong Mục 4.1.1 của bài báo fPINNs.

Bài toán tổng quát:

```text
(-Delta)^(alpha/2) u(x) = f(x), x in (0, 1)
u(0) = u(1) = 0
alpha = 1.5
```

Có hai trường hợp:

```text
smooth:
  u(x) = x^3(1-x)^3
  f(x) = 1 / (2 cos(pi alpha / 2)) * [
    Gamma(4) / Gamma(4-alpha) * (x^(3-alpha) + (1-x)^(3-alpha))
  - 3 Gamma(5) / Gamma(5-alpha) * (x^(4-alpha) + (1-x)^(4-alpha))
  + 3 Gamma(6) / Gamma(6-alpha) * (x^(5-alpha) + (1-x)^(5-alpha))
  - Gamma(7) / Gamma(7-alpha) * (x^(6-alpha) + (1-x)^(6-alpha))
  ]
  alpha = 1.5

nonsmooth:
  u(x) = x(1-x^2)^(alpha/2)
  f(x) = Gamma(alpha+2)x
  alpha = 1.5
```

## 1. Cấu trúc code

### `utils.py`

Chứa nghiệm đúng và hàm nguồn của bài toán.

Các thành phần chính:

```python
exact_solution_smooth(x)
forcing_term_smooth(x, alpha)
exact_solution_nonsmooth(x, alpha)
forcing_term_nonsmooth(x, alpha)
exact_solution_poisson(x)
forcing_term_poisson(x, alpha)
generate_training_data(N, alpha, smooth=True)
```

`smooth` sử dụng công thức hàm nguồn (4.3) trong bài báo. `nonsmooth` sử dụng hàm nguồn `Gamma(alpha+2)x`.

### `gl.py`

Sử dụng ma trận Grünwald–Letnikov (GL) cho toán tử Laplace phân số trên `[0, 1]`.

Các thành phần chính:

```python
build_gl_matrix(N, alpha, order)
get_gl_matrix(N, alpha, order)
```

Trong đó:

```text
N      : số khoảng lưới
alpha  : bậc phân số
order  : bậc GL, gồm 1, 2, 3
```

Đầu ra là ma trận:

```text
A_GL shape = (N-1, N+1)
```

Ma trận này được nhân với vector nghiệm trên toàn bộ lưới `[x_0, ..., x_N]` để xấp xỉ toán tử Laplace phân số tại các điểm nội miền `[x_1, ..., x_{N-1}]`.

### `network.py`

Chứa mạng fPINN và hàm mất mát.

Các thành phần chính:

```python
class fPINN
build_loss_poisson(...)
build_loss_poisson_lambda(...)
train(...)
```

Mạng trả về:

```text
u_hat(x) = x(1-x) NN(x)
```

Việc nhân với `x(1-x)` giúp áp đặt điều kiện biên:

```text
u_hat(0) = 0
u_hat(1) = 0
```

`build_loss_poisson(...)` dùng cho trường hợp `lambda=N`, như trong Hình 4 và Hình 5.

`build_loss_poisson_lambda(...)` dùng cho trường hợp `lambda` độc lập với `N`, như trong Hình 6 và Hình 7.

Hàm mất mát có dạng:

```text
MSE(A_GL u_hat - f)
```

### `error.py`

Chứa các độ đo sai số và hàm đánh giá mô hình.

Các thành phần chính:

```python
relative_l2_error(u_pred, u_exact)
mse_error(u_pred, u_exact)
max_abs_error(u_pred, u_exact)
evaluate_relative_l2(model, device, num_points, alpha, case)
```

`evaluate_relative_l2(...)` tự động chọn nghiệm đúng theo `case`:

```text
case="smooth"    -> u=x^3(1-x)^3
case="nonsmooth" -> u=x(1-x^2)^(alpha/2)
```

### `results_io.py`

Phụ trách đọc/ghi kết quả vào thư mục `Poisson/results`.

Các thành phần chính:

```python
write_csv(...)
read_csv(...)
write_config(...)
save_convergence_results(...)
save_fig5_trace(...)
save_fig6_summary(...)
save_fig7_summary(...)
```

Tệp này định nghĩa cấu trúc các tệp CSV:

```text
fig4_raw.csv
fig4_summary.csv
fig5_trace.csv
fig6_summary.csv
fig7_summary.csv
```

### `figures.py`

Tệp vẽ hình từ CSV.

Vai trò:

```text
1. Đọc CSV trong Poisson/results
2. Vẽ PNG
3. Không huấn luyện mô hình
4. Không ghi CSV
```

Đầu ra:

```text
fig4_convergence.png
fig5_loss_error.png
fig6_convergence.png
fig7_architecture.png
```

### `main.py`

Tệp thực thi các thí nghiệm.

Vai trò:

```text
1. Đọc tham số dòng lệnh
2. Tạo mô hình fPINN
3. Tạo hàm mất mát
4. Huấn luyện bằng Adam
5. Gọi error.py để tính sai số tương đối L2
6. Gọi results_io.py để ghi CSV/config
```

`main.py` không vẽ hình.

Nhóm thí nghiệm:

```text
fig4:
  bài toán smooth
  GL bậc 1, 2, 3
  lambda = N

fig5:
  bài toán smooth
  GL bậc 3
  N = 10 và N = 20
  vẽ loss/error theo số vòng lặp

fig6:
  bài toán nonsmooth
  panel (a): cố định N, quét lambda
  panel (b): cố định lambda, quét N

fig7:
  bài toán nonsmooth
  cố định N và lambda
  quét kiến trúc mạng width/depth
```

## 2. Cách chạy để tạo kết quả

Quy trình chung gồm hai bước:

```powershell
python Poisson\main.py ...
python Poisson\figures.py ...
```

`main.py` huấn luyện và tạo CSV. `figures.py` đọc CSV và vẽ PNG.

### Chạy bài toán smooth

Smooth case tạo Hình 4 và Hình 5.

```powershell
python Poisson\main.py --only smooth
python Poisson\figures.py --only smooth
```

Đầu ra:

```text
Poisson\results\fig4_raw.csv
Poisson\results\fig4_summary.csv
Poisson\results\fig5_trace.csv
Poisson\results\fig4_convergence.png
Poisson\results\fig5_loss_error.png
```

Chạy riêng Hình 4:

```powershell
python Poisson\main.py --only fig4 --orders 1 2 3
python Poisson\figures.py --only fig4
```

Chạy riêng Hình 5:

```powershell
python Poisson\main.py --only fig5 --fig5-ns 10 20
python Poisson\figures.py --only fig5
```

### Chạy bài toán nonsmooth

Nonsmooth case tạo Hình 6 và Hình 7.

```powershell
python Poisson\main.py --only nonsmooth
python Poisson\figures.py --only nonsmooth
```

Đầu ra:

```text
Poisson\results\fig6_summary.csv
Poisson\results\fig7_summary.csv
Poisson\results\fig6_convergence.png
Poisson\results\fig7_architecture.png
```

Chạy riêng Hình 6:

```powershell
python Poisson\main.py --only fig6
python Poisson\figures.py --only fig6
```

Chạy riêng Hình 7:

```powershell
python Poisson\main.py --only fig7
python Poisson\figures.py --only fig7
```

### Chạy tất cả

```powershell
python Poisson\main.py --cases smooth nonsmooth --only both
python Poisson\figures.py --only both
```

### Chạy nhanh để kiểm tra pipeline

Smooth nhanh:

```powershell
python Poisson\main.py --only smooth --ns 10 --orders 1 2 3 --seeds 0 --iterations 2 --fig5-ns 10 20 --fig5-iterations 2 --fig5-lr 1e-3 --eval-points 20 --widths 4 --depths 1
python Poisson\figures.py --only smooth
```

Nonsmooth nhanh:

```powershell
python Poisson\main.py --only nonsmooth --fig6-fixed-n 10 --fig6-lambdas 5 10 --fig6-fixed-lambda 10 --fig6-ns 5 10 --fig7-n 10 --fig7-lambda 10 --fig7-widths 4 5 --fig7-depths 1 2 --fig7-line-depths 1 2 3 --fig7-line-widths 4 5 6 --seeds 0 --iterations 2 --widths 4 --depths 1
python Poisson\figures.py --only nonsmooth
```

### Chạy gần với cấu hình trong bài báo

#### Hình 4

Bài báo sử dụng bài toán smooth, 3 bậc GL, 10 lần khởi tạo, và các cặp learning rate/số vòng lặp:

```text
1e-3 -> 1e5
1e-4 -> 1e6
1e-5 -> 1e7
1e-6 -> 1e7
```

Lệnh:

```powershell
python Poisson\main.py --only fig4 --orders 1 2 3 --learning-rates 1e-3:100000 1e-4:1000000 1e-5:10000000 1e-6:10000000 --seeds 0 1 2 3 4 5 6 7 8 9
python Poisson\figures.py --only fig4
```

Lệnh này sẽ chạy rất lâu.

#### Hình 5

Bài báo sử dụng bài toán smooth, GL bậc 3, `N=10` và `N=20`, learning rate `1e-6`, ghi lại quá trình huấn luyện đến `1e7` vòng lặp.

```powershell
python Poisson\main.py --only fig5 --fig5-ns 10 20 --fig5-lr 1e-6 --fig5-iterations 10000000
python Poisson\figures.py --only fig5
```

Lệnh này sẽ chạy rất lâu.

#### Hình 6

Bài báo sử dụng bài toán nonsmooth:

```text
panel (a): cố định N, quét lambda
panel (b): cố định lambda, quét N
```

Do chú thích hình không ghi learning rate riêng, có thể sử dụng giá trị mặc định của bài báo:

```text
learning rate = 5e-4
iterations = 1e5
```

Lệnh gợi ý:

```powershell
python Poisson\main.py --only fig6 --learning-rates 5e-4 --iterations 100000 --seeds 0 1 2 3 4 5 6 7 8 9 --fig6-fixed-n 100 --fig6-lambdas 20 40 80 160 320 500 --fig6-fixed-lambda 100 --fig6-ns 10 20 40 80 160 320 500
python Poisson\figures.py --only fig6
```

#### Hình 7

Bài báo sử dụng bài toán nonsmooth:

```text
N = 100
lambda = 200
learning rate = 1e-3
```

Lệnh gợi ý:

```powershell
python Poisson\main.py --only fig7 --learning-rates 1e-3 --iterations 100000 --seeds 0 1 2 3 4 5 6 7 8 9 --fig7-n 100 --fig7-lambda 200 --fig7-widths 20 30 40 50 60 --fig7-depths 2 3 4 5 6 7 8 --fig7-narrow-width 10 --fig7-line-depths 2 4 8 16 24 32 40 --fig7-shallow-depth 2 --fig7-line-widths 10 20 30 40 50 100 1000
python Poisson\figures.py --only fig7
```

Lệnh này sẽ chạy rất lâu.

## Thư mục kết quả

Tất cả kết quả nằm trong:

```text
Poisson\results
```

CSV:

```text
config.json
fig4_raw.csv
fig4_summary.csv
fig5_trace.csv
fig6_summary.csv
fig7_summary.csv
```

PNG:

```text
fig4_convergence.png
fig5_loss_error.png
fig6_convergence.png
fig7_architecture.png
```
