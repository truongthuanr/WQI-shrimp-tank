| Mô hình | Biến đầu vào | Tên trường hợp |
|---|---|---|
| ANN | Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm. | ANN-8 |
| ANN | Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm + kiềm dự đoán (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | ANN-8+ kiềm dự đoán |
| ANN | Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm + kiềm thực | ANN-8+ kiềm thực |
| ANN | Nhiệt độ, Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm | ANN-9 |
| ANN | Nhiệt độ, Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm + kiềm dự đoán (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | ANN-9+ kiềm dự đoán |
| ANN | Nhiệt độ, Giống tôm, area, TDS, Độ cứng, pH, Loại ao,Công nghệ nuôi, Tuổi tôm + kiềm thực | ANN-9+ kiềm thực |
| RF | Nhiệt độ, độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm | RF-9 |
| RF | Nhiệt độ, độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm + kiềm thực | RF-9- kiềm thực |
| RF | Nhiệt độ, độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm + kiềm dự báo (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | RF-9- kiềm dự báo |
| RF | Nhiệt độ, độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm + kiềm dự báo (bằng RF) | RF-9- kiềm dự báo (ít input) |
| RF | Độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm | RF-8 |
| RF | Độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm + kiềm thực | RF-8- kiềm thực |
| RF | Độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm  + kiềm dự báo (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | RF-8- kiềm dự báo |
| RF | Độ màu, area, độ mặn, loại ao, độ cứng, TDS, pH, Tuổi tôm  + kiềm dự báo (bằng RF) | RF-8- kiềm dự báo (ít input) |
| GBT | Độ trong, Độ cứng, Độ mặn, Nhiệt độ, TDS, Loại ao, Công nghệ nuôi, pH, Tuổi tôm | GBT-9 |
| GBT | Độ trong, Độ cứng, Độ mặn, Nhiệt độ, TDS, Loại ao, Công nghệ nuôi, pH, Tuổi tôm + kiềm dự báo (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | GBT-9-kiềm dự báo |
| GBT | Độ cứng, Độ mặn, Nhiệt độ, TDS, Loại ao, Công nghệ nuôi, pH, Tuổi tôm | GBT-8 |
| GBT | Độ cứng, Độ mặn, Nhiệt độ, TDS, Loại ao, Công nghệ nuôi, pH, Tuổi tôm + kiềm dự báo (bằng RF với season, loại ao, công nghệ, giống, ngày thả, diện tích ao, tuổi tôm, Nhiệt độ, pH, Độ mặn, mực nước, độ trong làm input) | GBT-9-kiềm dự báo |
| GBT | Độ trong', 'Độ cứng', 'Độ mặn', 'Nhiệt độ', 'TDS', 'Loại ao', 'Công nghệ nuôi', 'pH', 'Tuổi tôm', 'Độ kiềm' | GBT-9-kiềm thực |
| GBT | Độ cứng', 'Độ mặn', 'Nhiệt độ', 'TDS', 'Loại ao', 'Công nghệ nuôi', 'pH', 'Tuổi tôm', 'Độ kiềm' | GBT-8- kiềm thực |
