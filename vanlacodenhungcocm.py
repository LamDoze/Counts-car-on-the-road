import cv2  # Thư viện xử lý ảnh và video
import numpy as np  # Thư viện xử lý các phép tính toán ma trận
import tkinter as tk  # Thư viện tạo giao diện đồ họa
from tkinter import Label, Frame  # Các thành phần giao diện từ tkinter
from PIL import Image, ImageTk  # Thư viện để chuyển đổi và hiển thị ảnh trong tkinter

# Thiết lập các tham số cho việc phát hiện phương tiện
min_contour_width = 50  # Chiều rộng nhỏ nhất của contour để được xem là một phương tiện
min_contour_height = 50  # Chiều cao nhỏ nhất của contour để được xem là một phương tiện
offset = 10  # Sai số cho phép khi tâm đi qua đường kẻ
line_height = 550  # Vị trí của đường kiểm tra (theo trục y)
matches = []  # Danh sách chứa tâm của các phương tiện đã phát hiện
cars = 0  # Bộ đếm số lượng phương tiện

def get_centroid(x, y, w, h):
    """
    Hàm tính toán tâm (centroid) của một hình chữ nhật dựa trên tọa độ góc trên-trái và kích thước.
    """
    cx = int(x + w / 2)  # Tọa độ x của tâm
    cy = int(y + h / 2)  # Tọa độ y của tâm
    return cx, cy  # Trả về tọa độ tâm

# Mở nguồn video
cap = cv2.VideoCapture('../Videos/traffic.mp4')

def process_frame():
    """
    Hàm xử lý từng khung hình video, phát hiện phương tiện, và cập nhật giao diện hiển thị.
    """
    global cars, frame1, frame2

    ret, frame2 = cap.read()  # Đọc khung hình tiếp theo từ video
    if not ret:  # Nếu không đọc được khung hình
        cap.release()  # Giải phóng tài nguyên video
        window.destroy()  # Đóng cửa sổ giao diện
        return

    # Xử lý khung hình
    d = cv2.absdiff(frame1, frame2)  # Tính độ chênh lệch giữa 2 khung hình liên tiếp
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang thang độ xám
    blur = cv2.GaussianBlur(grey, (5, 5), 0)  # Làm mờ ảnh bằng Gaussian Blur
    edges = cv2.Canny(blur, 50, 150)  # Tìm biên bằng thuật toán Canny
    _, th = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)  # Ngưỡng hóa ảnh (binary)

    # Tạo kernel để thực hiện các phép biến đổi hình thái học
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(th, kernel, iterations=2)  # Mở rộng các vùng trắng
    eroded = cv2.erode(dilated, kernel, iterations=1)  # Thu hẹp các vùng trắng
    closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)  # Đóng các lỗ hổng nhỏ bên trong

    # Sử dụng flood-fill để lấp đầy các lỗ trống bên trong
    filled_closing = closing.copy()
    h, w = filled_closing.shape[:2]  # Lấy kích thước của khung hình
    mask = np.zeros((h+2, w+2), np.uint8)  # Tạo mặt nạ cho flood-fill 
    #Kích thước này được dùng để tạo một "vùng biên" xung quanh ảnh xử lý nhằm tránh lỗi khi flood-fill chạm tới rìa của ảnh.
    #Kích thước mặt nạ sẽ trở thành (h+2, w+2):
    #Nếu ảnh gốc có kích thước 600x800, kích thước mặt nạ sẽ là 602x802.
    cv2.floodFill(filled_closing, mask, (0, 0), 255)  # Lấp đầy từ góc trên-trái
    filled_closing = cv2.bitwise_not(filled_closing)  # Đảo ngược màu
    filled_closing = cv2.bitwise_or(closing, filled_closing)  # Kết hợp với ảnh gốc

    # Tìm các contour trong ảnh
    contours, _ = cv2.findContours(filled_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Xử lý từng contour
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)  # Lấy hình chữ nhật bao quanh contour
        if w >= min_contour_width and h >= min_contour_height and w * h >= 2500:  # Kiểm tra kích thước tối thiểu
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật quanh phương tiện
            centroid = get_centroid(x, y, w, h)  # Tính tâm của contour
            matches.append(centroid)  # Lưu tâm vào danh sách
            cx, cy = centroid
            if (line_height - offset) < cy < (line_height + offset):  # Kiểm tra nếu tâm nằm trong vùng đường kiểm tra
                cars += 1  # Tăng bộ đếm số lượng xe
                print(f"Số lượng xe đã được đếm: {cars}") 
                if (cx, cy) in matches:  # Xóa tâm đã xử lý để tránh đếm lại
                    matches.remove((cx, cy))

    # Vẽ đường kiểm tra và hiển thị số lượng xe
    cv2.line(frame1, (100, line_height), (900, line_height), (0, 255, 0), 2)
    cv2.putText(frame1, f"Tong so xe: {cars}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    # Resize khung hình để hiển thị trên giao diện
    resized_frame1 = cv2.resize(frame1, (800, 600))  # Thay đổi kích thước khung hình gốc
    resized_frame2 = cv2.resize(filled_closing, (800, 600))  # Thay đổi kích thước ảnh nhị phân

    # Hiển thị khung hình gốc
    img1 = Image.fromarray(cv2.cvtColor(resized_frame1, cv2.COLOR_BGR2RGB))  # Chuyển đổi sang định dạng RGB
    imgtk1 = ImageTk.PhotoImage(image=img1)  # Chuyển đổi sang định dạng ImageTk
    label1.imgtk = imgtk1
    label1.configure(image=imgtk1)

    # Hiển thị khung hình đã xử lý
    img2 = Image.fromarray(resized_frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    label2.imgtk = imgtk2
    label2.configure(image=imgtk2)

    # Cập nhật khung hình
    frame1 = frame2
    window.after(10, process_frame)  # Lặp lại hàm sau 10ms

# Tạo cửa sổ GUI với bố cục hiển thị song song
window = tk.Tk()
window.title("Vehicle Detection System")  # Tiêu đề cửa sổ
window.geometry("1700x700")  # Kích thước cửa sổ chính

# Tạo các khung hiển thị song song
frame_left = Frame(window, width=800, height=600, bg="black")  # Khung hiển thị bên trái
frame_left.pack(side="left", padx=20, pady=20)

frame_right = Frame(window, width=800, height=600, bg="black")  # Khung hiển thị bên phải
frame_right.pack(side="right", padx=20, pady=20)

# Nhãn để hiển thị video
label1 = Label(frame_left)
label1.pack()
label2 = Label(frame_right)
label2.pack()

# Bắt đầu đọc video
ret, frame1 = cap.read()  # Đọc khung hình đầu tiên
if ret:
    process_frame()  # Bắt đầu xử lý video

# Giải phóng tài nguyên khi đóng cửa sổ
window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), window.destroy()))
window.mainloop()
