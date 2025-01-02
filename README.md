🌟 Giới thiệu

Dự án này là một ứng dụng đơn giản giúp đếm số lượng xe ô tô di chuyển qua một khu vực cụ thể trong video. Ứng dụng được xây dựng bằng cách kết hợp OpenCV và Tkinter, hiển thị video và các bước xử lý ảnh theo thời gian thực.

🛠️ Chức năng chính

  Xử lý video: Đọc và phân tích video giao thông.
  Phát hiện chuyển động: Dùng kỹ thuật phát hiện sự khác biệt giữa các khung hình để tìm xe.
  Đếm xe: Phát hiện khi xe đi qua một đường kẻ xác định.
  Hiển thị giao diện trực quan:
      Video gốc có hiển thị các khung chữ nhật quanh xe.
      Video sau khi xử lý các bước như làm mờ, phát hiện cạnh, và lọc nhiễu.

  ![image](https://github.com/user-attachments/assets/fdf71225-59f9-4610-8069-991498e8f06d)

🧐 Cách hoạt động

  Đọc video: Mỗi khung hình sẽ được so sánh với khung hình trước đó để phát hiện sự thay đổi.
  Xử lý ảnh:
      Làm mờ để giảm nhiễu.
      Phát hiện cạnh và lọc những vùng nhỏ không phải xe.
      Đóng vùng và tìm contour.
  Phát hiện xe: Nếu contour đủ lớn và vượt qua đường kẻ xanh, xe sẽ được đếm.
  Hiển thị kết quả:
      Số lượng xe được cập nhật trên khung hình.
      Hai video được hiển thị song song:
          Video gốc với các khung hình chữ nhật quanh xe.
          Video đã qua xử lý.
          
🚀 Học hỏi từ dự án

  Làm quen với OpenCV để xử lý ảnh và video.
  Sử dụng Tkinter để tạo giao diện người dùng.
  Áp dụng các kỹ thuật xử lý ảnh như làm mờ, phát hiện cạnh, và đóng vùng.

  Khi áp dụng vào một nơi đông đúc xe cộ hơn thì nó vẫn còn khá nhiều yếu điểm 
  VD: Bộ đếm không chính xác, các xe máy đi sát gần nhau thì nó vẫn tính thành 1 oto, xe bus quá lớn để có thể nhận diện, v..v...

  ![image](https://github.com/user-attachments/assets/75621ed9-c36f-4cd3-8eec-74121c685dc0)

❤️ Cảm ơn

Nếu bạn thấy dự án này hữu ích, hãy nhấn ⭐ để ủng hộ nhé! 🚀

