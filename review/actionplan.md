Về vấn đề input (feature selection):

1. Mục tiêu
- Loại bỏ mâu thuẫn giữa phần correlation/importance và bộ input dùng cho kết quả chính.
- Đảm bảo quy trình chọn biến có thể tái lập, không leakage, trả lời được reviewer.

2. Nguyên tắc bắt buộc
- Chọn model chính trước trên bộ biến `core` cố định (ANN/SVR/RFR dùng cùng input để so sánh công bằng).
- Giảm biến chỉ thực hiện sau khi đã chọn model chính.
- Không dùng test set để chọn biến.

3. Quy trình thực hiện
- Bước 1: Khóa bộ biến `core` theo dữ liệu + domain knowledge và ghi rõ trong Methods.
- Bước 2: Huấn luyện/chọn model chính bằng CV trên train (giữ nguyên `core`).
- Bước 3: Trên model chính, tính feature importance trên train folds:
  - RF impurity importance
  - Permutation importance (bắt buộc để kiểm chứng)
- Bước 4: Tạo các bộ biến rút gọn theo quy tắc cố định:
  - Top-k (ví dụ k = 6, 8, 10)
  - Hoặc ngưỡng importance + tiêu chí ổn định qua folds/seeds
- Bước 5: Chạy ablation có kiểm soát cho từng bộ rút gọn:
  - So sánh RMSE, MAE, R2, Bias và độ lệch chuẩn CV
- Bước 6: Chọn bộ biến final:
  - Ưu tiên bộ ít biến nhất nhưng hiệu năng không giảm đáng kể và ổn định

4. Tiêu chí quyết định
- Bộ rút gọn được chấp nhận nếu:
  - RMSE không xấu hơn đáng kể so với `core`
  - R2 không giảm đáng kể
  - Kết quả ổn định qua nhiều fold/seed
- Nếu không đạt: giữ lại bộ `core` làm kết quả chính.

5. Nội dung cần cập nhật vào manuscript/rebuttal
- Thêm bảng: "Feature sets by experiment stage" (core vs reduced).
- Thêm bảng: "Ablation results" với đầy đủ metric.
- Viết rõ: correlation chỉ mang tính mô tả; tiêu chí chọn biến là CV performance + stability.
- Nêu rõ đánh giá chính dùng bộ nào, giảm biến là phân tích bổ sung hay final.

6. Deliverables
- `feature_set_table.csv` (danh sách biến theo từng stage)
- `ablation_summary.csv` (metric cho core và từng bộ reduced)
- 1 đoạn Methods cập nhật + 1 đoạn Response to Reviewer cho issue input selection



External validation (cross-site): site-specific retraining strategy

1. Mục tiêu
- Chuyển claim từ "robustness trên independent datasets" sang diễn giải thận trọng hơn: "khả năng thích nghi tốt sau local recalibration".

- Trả lời reviewer bằng số liệu minh bạch theo từng site.

2. Bối cảnh kết quả hiện tại
- Đã thử zero-shot (train site gốc -> predict site khác) và kết quả không tốt.
- Kết luận kỹ thuật: có domain shift giữa các site nên cần retraining.

3. Giải thích khác biệt đặc tính site 
- Khác biệt thiết bị đo và quy trình lấy mẫu.
- Khác biệt điều kiện nuôi/vận hành ao.
- Khác biệt phân phối biến đầu vào và dải alkalinity.
- Các lý do tham khảo khác có thể gây khác biệt giữa site:
  - Khác nguồn nước đầu vào (độ kiềm nền, độ mặn nền, tải hữu cơ nền).
  - Khác thiết kế ao (diện tích, độ sâu, lót bạt/đất, hệ lắng-lọc).
  - Khác chế độ thay nước (tần suất, tỷ lệ, thời điểm thay).
  - Khác mật độ thả, cỡ giống và giai đoạn vụ nuôi.
  - Khác chiến lược cho ăn và FCR.
  - Khác vận hành sục khí/trộn nước.
  - Khác lịch bổ sung khoáng, vôi, bicarbonate, probiotic.
  - Khác điều kiện thời tiết vi mô theo vùng và mùa.
  - Khác tình trạng đáy ao và hoạt động vi sinh.

4. Protocol đánh giá bắt buộc
- Protocol A (Zero-shot transfer):
  - Train trên site gốc, giữ nguyên pipeline/hyperparameters, test trực tiếp site mới.
- Protocol B (Site-specific retraining):
  - Train/tune lại trên dữ liệu train của từng site mới.
  - Đánh giá trên holdout nội bộ của chính site đó.

5. Quy tắc chống leakage
- Preprocessing/scaler fit trên train của từng protocol, không fit trên toàn bộ dữ liệu site.
- Test set chỉ dùng để đánh giá cuối.

6. Bảng kết quả cần bổ sung
- Bảng mô tả dữ liệu theo site: `n`, min, max, mean, SD của biến chính.
- Bảng hiệu năng theo site cho cả 2 protocol: `R2`, `RMSE`, `MAE`, `Bias`.
- Thêm cột `delta` (mức cải thiện của retraining so với zero-shot).

7. Cập nhật claim trong manuscript
- Methods: thêm mục "Transfer vs Adaptation evaluation protocol".
- Results: trình bày rõ zero-shot kém, retraining cải thiện.
- Discussion/Conclusion: nhấn mạnh framework có tính ứng dụng nhờ khả năng adapt theo site; điều chỉnh các câu đang nói "robustness trên independent datasets" để không hàm ý khái quát trực tiếp cho mọi site.

8. Nội dung rebuttal cần trả reviewer
- Nêu thẳng kết quả zero-shot không đủ mạnh do khác biệt đặc tính site.
- Cung cấp số liệu site-wise sau retraining để chứng minh tính khả dụng thực tế.

