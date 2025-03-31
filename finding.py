import os
import cv2
import numpy as np
from collections import Counter
from PIL import Image

def is_near_white(color, threshold=240):
    """Kiểm tra xem màu có gần trắng không (giúp loại bỏ nền trắng)"""
    return all(c >= threshold for c in color)

def find_most_common_color(template_path):
    """
    Đọc ảnh template, loại bỏ các pixel gần trắng và chọn màu phổ biến nhất.
    Kết quả trả về là màu ở định dạng BGR.
    """
    image = Image.open(template_path).convert('RGBA')
    data = np.array(image)
    mask = data[:, :, 3] > 0  # loại bỏ vùng trong suốt
    colors = data[mask][:, :3]
    color_counts = Counter(map(tuple, colors))
    sorted_colors = [color for color, _ in color_counts.most_common() if not is_near_white(color)]
    if sorted_colors:
        # Chuyển từ RGB sang BGR
        return sorted_colors[0][::-1]
    else:
        return None

def extract_similar_color_regions(image_path, template_color, color_tolerance=30, dilation_size=20):
    """
    Dựa vào template_color, tìm các vùng có màu tương tự trên ảnh lớn.
    - Tính khoảng cách giữa mỗi pixel và template_color.
    - Tạo mask cho những pixel có khoảng cách dưới color_tolerance.
    - Mở rộng mask (dilation) để bao quát vùng liên quan.
    - Lấy ra các vùng đó từ ảnh gốc.
    """
    image = cv2.imread(image_path)  # Ảnh gốc ở dạng BGR
    diff = np.linalg.norm(image.astype(np.float32) - np.array(template_color, dtype=np.float32), axis=-1)
    mask = (diff < color_tolerance).astype(np.uint8) * 255

    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    extracted = cv2.bitwise_and(image, image, mask=mask_dilated)
    return extracted, mask_dilated

def smooth_image(image, method="gaussian", kernel_size=5):
    """
    Làm mượt ảnh trước khi phát hiện cạnh.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("Phương pháp làm mượt không hợp lệ!")

def convert_to_edges(image, low_threshold=50, high_threshold=150, kernel_size=5, closing_iterations=1):
    """
    Chuyển đổi ảnh màu sang ảnh xám, làm mịn ảnh, sau đó dùng Canny để lấy cạnh.
    Áp dụng morphological closing để kết nối các đoạn cạnh bị gián đoạn.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=10, sigmaSpace=75)
    edges = cv2.Canny(filtered, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    return edges_closed

def multi_scale_template_matching_on_edges(large_edge, template_edge, scale_range=(0.5, 1.5), scale_step=0.1):
    """
    Áp dụng multi-scale template matching trên ảnh edge.
    Trả về tọa độ match (góc trên trái và góc dưới phải) trên ảnh gốc và giá trị match tốt nhất.
    """
    best_match_val = -1
    best_scale = 1.0
    best_top_left = None
    template_w, template_h = template_edge.shape[::-1]

    for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
        resized_large = cv2.resize(large_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if resized_large.shape[0] < template_edge.shape[0] or resized_large.shape[1] < template_edge.shape[1]:
            continue
        result = cv2.matchTemplate(resized_large, template_edge, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_match_val:
            best_match_val = max_val
            best_scale = scale
            best_top_left = max_loc

    if best_top_left is None:
        return None, None, None

    original_top_left = (int(best_top_left[0] / best_scale), int(best_top_left[1] / best_scale))
    original_bottom_right = (original_top_left[0] + template_w, original_top_left[1] + template_h)
    
    print(f"Best match value: {best_match_val:.2f} tại scale: {best_scale:.2f}")
    return original_top_left, original_bottom_right, best_match_val

# Đường dẫn ảnh lớn cần tìm đối tượng
large_image_path = "1.jpg"
large_image = cv2.imread(large_image_path)

# Đường dẫn tới thư mục chứa nhiều template
templates_folder = "template"

# Danh sách file template (có thể giới hạn đuôi mở rộng nếu cần)
template_files = [os.path.join(templates_folder, f) for f in os.listdir(templates_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Danh sách để lưu các kết quả bounding box
bounding_boxes = []

# Duyệt qua từng template trong thư mục
for template_path in template_files:
    print("\nĐang xử lý template:", template_path)
    # Bước 1: Tìm màu phổ biến nhất từ template
    template_common_color = find_most_common_color(template_path)
    if template_common_color is None:
        print("Không tìm thấy màu hợp lệ trong template:", template_path)
        continue
    print("Màu phổ biến (BGR):", template_common_color)
    
    # Bước 2: Tìm vùng có màu tương tự trên ảnh lớn
    extracted_image, _ = extract_similar_color_regions(
        large_image_path, template_common_color,
        color_tolerance=3, dilation_size=40
    )
    
    # Bước 3: Chuyển đổi ảnh trích xuất thành ảnh edge
    edge_extracted = convert_to_edges(
        extracted_image, low_threshold=50, high_threshold=150, closing_iterations=2
    )
    
    # Bước 4: Chuyển đổi template thành ảnh edge
    template_image = cv2.imread(template_path)
    template_edge = convert_to_edges(
        template_image, low_threshold=50, high_threshold=150, closing_iterations=2
    )
    
    # Bước 5: Áp dụng multi-scale template matching
    top_left, bottom_right, match_val = multi_scale_template_matching_on_edges(
        edge_extracted, template_edge, scale_range=(0.5, 1.5), scale_step=0.1
    )
    
    if top_left is None:
        print("Không tìm thấy đối tượng phù hợp cho template:", template_path)
        continue
    # Lưu bounding box cùng với tên template và giá trị match
    bounding_boxes.append((top_left, bottom_right, os.path.basename(template_path), match_val))

# Vẽ tất cả bounding box lên ảnh gốc
for (top_left, bottom_right, template_name, match_val) in bounding_boxes:
    cv2.rectangle(large_image, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(large_image, f"{template_name}:{match_val:.2f}", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("final_matched_aggregated.png", large_image)
print("\n Lưu ảnh tổng hợp với các bounding box: final_matched_aggregated.png")
