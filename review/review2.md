### Bảng 3.9 Kết quả mô phỏng khi giảm dần số biến đầu vào

| Trường hợp | Mô hình | R2 |RMSE | MAE |
|---|---|---|---|---|
| Giữ lại 10 thông số | RF | 0.75 | 12.93 | 9.11 |
| Giữ lại 10 thông số | GBT | 0.73 | 13.07 |
| Giữ lại 10 thông số | ANN | 0.63 | 15.17 |
| Giữ lại 10 thông số | SVR | 0.60 | 15.31 |
| Giữ lại 9 thông số | RF | 0.73 | 12.91 |
| Giữ lại 9 thông số | GBT | 0.73 | 13.06 |
| Giữ lại 9 thông số | ANN | 0.63 | 15.14 |
| Giữ lại 9 thông số | SVR | 0.59 | 15.62 |
| Giữ lại 8 thông số | RF | 0.73 | 12.95 |
| Giữ lại 8 thông số | GBT | 0.72 | 13.27 |
| Giữ lại 8 thông số | ANN | 0.64 | 15.07 |
| Giữ lại 8 thông số | SVR | 0.58 | 15.79 |
| Giữ lại 7 thông số | RF | 0.71 | 13.38 |
| Giữ lại 7 thông số | GBT | 0.71 | 13.54 |
| Giữ lại 7 thông số | ANN | 0.63 | 15.16 |
| Giữ lại 7 thông số | SVR | 0.58 | 15.78 |
| Giữ lại 6 thông số | RF | 0.71 | 13.47 |
| Giữ lại 6 thông số | GBT | 0.71 | 13.43 |
| Giữ lại 6 thông số | ANN | 0.63 | 15.16 |
| Giữ lại 6 thông số | SVR | 0.48 | 17.63 |
| Giữ lại 5 thông số | RF | 0.72 | 13.26 |
| Giữ lại 5 thông số | GBT | 0.71 | 13.50 |
| Giữ lại 5 thông số | ANN | 0.61 | 15.56 |
| Giữ lại 5 thông số | SVR | 0.48 | 17.57 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | RF | 0.71 | 13.53 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | GBT | 0.71 | 13.35 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | ANN | 0.55 | 16.69 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | SVR | 0.53 | 16.64 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | RF | 0.74 | 11.57 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | GBT | 0.73 | 12.44 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | ANN | 0.69 | 11.25 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | SVR | 0.62 | 15.48 |

Ghi chú hàng RF giữ lại 10 thông số: chạy lại với Tuổi tôm, Nhiệt độ, pH, Độ mặn, Mực nước, Công nghệ nuôi, Loại ao, Season, area và Độ trong; dự đoán Độ kiềm cùng thời điểm. Dữ liệu `data_4perday_cleaned.csv`, lọc z-score 3, chia ngẫu nhiên 60/20/20 với seed 42, đánh giá trên 601 mẫu test. R² = 0.7545, RMSE = 12.9272, MAE = 9.1078 (giá trị trong bảng làm tròn 2 chữ số). Hai giá trị R² = 0.73 và RMSE = 13.01 ghi trước đó không tái lập được với thiết lập này nên đã được thay bằng kết quả cùng lần chạy với MAE.

### Bảng 3.12 Kết quả dự báo kiềm của các kịch bản khác nhau với mô hình cải tiến

| Mô hình |Buổi trưa: R2 | Buổi trưa: RMSE | Buổi trưa: MAE | Buổi chiều: R2 | Buổi chiều: RMSE | Buổi chiều: MAE | Ngày +1: R2 |Ngày +1: RMSE | Ngày +1: MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN-8+ kiềm thực | | | | | | | 0.70 | 15.21 |
| ANN-8 + kiềm mô phỏng | | | | | | | 0.75 | 13.64 |
| ANN-9+ kiềm thực | | | | | | | 0.70 | 15.15 |
| ANN-9+ kiềm mô phỏng | | | | | | | 0.74 | 13.99 |
| RF-8 + kiềm thực | | | | | | | 0.74 | 14.22 |
| RF-8 + kiềm mô phỏng | | | | | | | 0.77 | 13.40 |
| RF-9+ kiềm thực | | | | | | | 0.74 | 14.20 |
| RF-9 + kiềm mô phỏng | | | | | | | 0.77 | 13.43 |
| GBT-8+ kiềm thực | | | | | | | 0.72 | 14.67 |
| GBT-8+ kiềm mô phỏng | | | | | | | 0.74 | 14.14 |
| GBT-9+ kiềm thực | | | | | | | 0.72 | 14.66 |
| GBT-9+ kiềm mô phỏng | | | | | | | 0.74 | 14.14 |

Các kết quả của mô hình  gồm kiềm thực và kiềm dự báo để minh chứng con số của mình là thật
