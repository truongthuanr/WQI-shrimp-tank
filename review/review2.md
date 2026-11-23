### Bảng 3.9 Kết quả mô phỏng khi giảm dần số biến đầu vào

| Trường hợp | Mô hình | R2 |RMSE | MAE |
|---|---|---|---|---|
| Giữ lại 10 thông số | RF | 0.73 | 13.01 | 9.11 |
| Giữ lại 10 thông số | GBT | 0.73 | 13.07 | 10.76 |
| Giữ lại 10 thông số | ANN | 0.63 | 15.17 | 10.44 |
| Giữ lại 10 thông số | SVR | 0.60 | 15.31 | 11.09 |
| Giữ lại 9 thông số | RF | 0.73 | 12.91 | 8.79 |
| Giữ lại 9 thông số | GBT | 0.73 | 13.06 | 11.12 |
| Giữ lại 9 thông số | ANN | 0.63 | 15.14 | 10.36 |
| Giữ lại 9 thông số | SVR | 0.59 | 15.62 | 10.10 |
| Giữ lại 8 thông số | RF | 0.73 | 12.95 | 8.56 |
| Giữ lại 8 thông số | GBT | 0.72 | 13.27 | 11.06 |
| Giữ lại 8 thông số | ANN | 0.64 | 15.07 | 10.02 |
| Giữ lại 8 thông số | SVR | 0.58 | 15.79 | 10.22 |
| Giữ lại 7 thông số | RF | 0.71 | 13.38 | 10.21 |
| Giữ lại 7 thông số | GBT | 0.71 | 13.54 | 12.46 |
| Giữ lại 7 thông số | ANN | 0.63 | 15.16 | 11.77 |
| Giữ lại 7 thông số | SVR | 0.58 | 15.78 | 12.77 |
| Giữ lại 6 thông số | RF | 0.71 | 13.47 | 11.27 |
| Giữ lại 6 thông số | GBT | 0.71 | 13.43 | 13.33 |
| Giữ lại 6 thông số | ANN | 0.63 | 15.16 | 12.51 |
| Giữ lại 6 thông số | SVR | 0.48 | 17.63 | 13.23 |
| Giữ lại 5 thông số | RF | 0.72 | 13.26 |
| Giữ lại 5 thông số | GBT | 0.71 | 13.50 |
| Giữ lại 5 thông số | ANN | 0.61 | 15.56 |
| Giữ lại 5 thông số | SVR | 0.48 | 17.57 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | RF | 0.71 | 13.53 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | GBT | 0.71 | 13.35 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | ANN | 0.55 | 16.69 |
| Giữ lại 10 thông số đóng góp và loại bớt các thông số khó đo | SVR | 0.53 | 16.64 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | RF | 0.74 | 11.57 | 9.11 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | GBT | 0.73 | 12.44 | 10.76 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | ANN | 0.69 | 11.25 | 10.44 |
| Chỉ giữ lại các thông số dễ đo (Tuổi tôm, nhiệt độ, pH, độ mặn, mực nước, độ trong, công nghệ nuôi, loại ao nuôi, mùa vụ và diện tích ao) | SVR | 0.62 | 15.48 | 11.09 |

Ghi chú: MAE của nhóm 8/9 biến được tính theo các bộ biến trong `variable-selection.md`, với SVR dùng bộ biến RF-8/9 tương ứng; dự đoán Độ kiềm cùng thời điểm (`shift=0`), chia ngẫu nhiên seed 42 và dùng chung 598 mẫu test. Nhóm 7 biến dùng Loại ao, Công nghệ nuôi, Độ cứng, Độ mặn, TDS, pH và Tuổi tôm; cùng thiết lập `shift=0`, seed 42 và 691 mẫu test. Nhóm 6 biến dùng Nhiệt độ, Độ cứng, pH, Công nghệ nuôi, Loại ao và Tuổi tôm; cùng thiết lập `shift=0`, seed 42 và 691 mẫu test. MAE ở bốn hàng thông số dễ đo được lấy từ lần chạy cùng bộ 10 biến ở trên (`shift=0`, 601 mẫu test). R² và RMSE được giữ nguyên theo bảng hiện có, không phải kết quả của các lần chạy MAE này.

### Bảng 3.12 Kết quả dự báo kiềm của các kịch bản khác nhau với mô hình cải tiến

| Mô hình |Buổi trưa: R2 | Buổi trưa: RMSE | Buổi trưa: MAE | Buổi chiều: R2 | Buổi chiều: RMSE | Buổi chiều: MAE | Ngày +1: R2 |Ngày +1: RMSE | Ngày +1: MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN-8+ kiềm thực | 0.75 | 14.06 | 8.96 | 0.77 | 15.25 | 9.65 | 0.70 | 15.21 | 13.02 |
| ANN-8 + kiềm mô phỏng | 0.65 | 16.75 | 11.60 | 0.80 | 14.29 | 10.33 | 0.75 | 13.64 | 12.74 |
| ANN-9+ kiềm thực | 0.73 | 14.79 | 9.34 | 0.77 | 15.33 | 9.65 | 0.70 | 15.15 | 12.55 |
| ANN-9+ kiềm mô phỏng | 0.62 | 17.44 | 11.82 | 0.78 | 15.12 | 10.90 | 0.74 | 13.99 | 12.97 |
| RF-8 + kiềm thực | 0.78 | 13.17 | 8.39 | 0.81 | 13.87 | 8.82 | 0.74 | 14.22 | 11.59 |
| RF-8 + kiềm mô phỏng | 0.69 | 15.87 | 10.88 | 0.81 | 13.97 | 10.34 | 0.77 | 13.40 | 12.03 |
| RF-9+ kiềm thực | 0.79 | 12.88 | 8.21 | 0.82 | 13.57 | 8.90 | 0.74 | 14.20 | 11.50 |
| RF-9 + kiềm mô phỏng | 0.69 | 15.69 | 10.68 | 0.82 | 13.64 | 10.24 | 0.77 | 13.43 | 12.00 |
| GBT-8+ kiềm thực | 0.82 | 12.04 | 7.48 | 0.78 | 15.04 | 9.57 | 0.72 | 14.67 | 13.21 |
| GBT-8+ kiềm mô phỏng | 0.67 | 16.27 | 11.59 | 0.75 | 16.11 | 11.74 | 0.74 | 14.14 | 14.16 |
| GBT-9+ kiềm thực | 0.81 | 12.22 | 7.76 | 0.71 | 17.22 | 9.72 | 0.72 | 14.66 | 12.57 |
| GBT-9+ kiềm mô phỏng | 0.65 | 16.63 | 11.82 | 0.72 | 16.86 | 11.51 | 0.74 | 14.14 | 13.65 |

Ghi chú Bảng 3.12: Bản ghi sáng trước 09:00 là đầu vào; nhãn buổi trưa là bản ghi cuối từ 09:00 đến trước 15:00, nhãn buổi chiều là bản ghi cuối từ 15:00 trở đi, và nhãn Ngày +1 là bản ghi sáng đúng ngày lịch kế tiếp. Kiềm thực là số đo ở bản ghi sáng; kiềm mô phỏng do RF dự đoán từ các biến khác của bản ghi sáng, với dự đoán ngoài mẫu cho tập huấn luyện và tập test. Số mẫu test tương ứng là 156, 118 và 186. R²/RMSE/MAE buổi trưa và buổi chiều cùng từ lần chạy mới; **Ngày +1 chỉ MAE từ lần chạy mới**, R²/RMSE giữ theo bảng cũ nên không được diễn giải chung như một bộ chỉ số của cùng lần đánh giá. Vì vậy hàng GBT-8 kiềm mô phỏng có MAE 14.16 lớn hơn RMSE cũ 14.14.

Các kết quả của mô hình  gồm kiềm thực và kiềm dự báo để minh chứng con số của mình là thật
