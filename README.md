# Object Detection Using Color and Edge Matching

## 📌 Giới thiệu
Dự án này áp dụng phương pháp phát hiện đối tượng dựa vào màu sắc và cạnh. Hệ thống sẽ tìm màu phổ biến trong template, xác định các vùng có màu tương tự trên ảnh lớn, sau đó sử dụng kỹ thuật phát hiện cạnh và template matching để tìm vị trí đối tượng.

## 🛠 Yêu cầu hệ thống
Trước khi chạy dự án, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết bằng cách chạy lệnh sau:

```bash
pip install numpy opencv-python pillow
```

## 🚀 Cách sử dụng
### 1️⃣ Chuẩn bị dữ liệu
- **Ảnh đầu vào**: Đặt ảnh lớn cần tìm đối tượng vào thư mục chứa script, với tên `1.jpg` (hoặc sửa trong code nếu cần).
- **Templates**: Đặt các ảnh template trong thư mục `template/`. Hỗ trợ các định dạng `.png`, `.jpg`, `.jpeg`.

### 2️⃣ Chạy chương trình
Thực thi script Python:

```bash
python finding.py
```

### 3️⃣ Kết quả
Sau khi chạy, chương trình sẽ:
- 📌 Phát hiện các đối tượng trong ảnh lớn.
- 🖼️ Vẽ bounding box quanh các đối tượng tìm thấy.
- 💾 Xuất ảnh kết quả với tên `final_matched_aggregated.png`.

## ⚙️ Giải thích thuật toán
1. **Xác định màu phổ biến nhất trong template** (loại trừ màu trắng hoặc gần trắng).
2. **Tìm các vùng có màu tương tự trên ảnh lớn** bằng cách so sánh khoảng cách màu.
3. **Làm mượt ảnh và phát hiện cạnh** bằng phương pháp Canny Edge Detection.
4. **Áp dụng Multi-Scale Template Matching** để tìm vị trí khớp nhất.
5. **Vẽ bounding box** xung quanh đối tượng phát hiện được.

## 🔧 Ghi chú
- Bạn có thể điều chỉnh các tham số như `color_tolerance`, `dilation_size`, `low_threshold`, `high_threshold` để cải thiện độ chính xác.
- Nếu muốn dùng ảnh lớn khác, hãy thay thế `1.jpg` trong thư mục và đảm bảo đường dẫn đúng trong script.

