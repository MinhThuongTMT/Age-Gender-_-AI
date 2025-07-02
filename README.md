# Dự án Phát hiện Tuổi, Giới tính và Cảm xúc qua Camera

Chào mừng đến với dự án sử dụng AI để nhận diện tuổi, giới tính và cảm xúc từ camera máy tính! Dự án này sử dụng thư viện DeepFace để phân tích hình ảnh thời gian thực, giúp bạn khám phá cách AI có thể áp dụng trong cuộc sống hàng ngày.

## Giới thiệu
Dự án này cho phép bạn:
- Phát hiện tuổi của người dùng.
- Xác định giới tính (Man hoặc Nữ) với độ chính xác cao.
- Phân tích cảm xúc dominant kèm phần trăm.

Dựa trên OpenCV và DeepFace, đây là một công cụ đơn giản nhưng mạnh mẽ để học hỏi về nhận diện khuôn mặt.

## Yêu cầu hệ thống
- Python 3.6 hoặc cao hơn.
- Camera máy tính (webcam).
- Các thư viện: opencv-python và deepface.

## Cài đặt
1. Clone repository hoặc đảm bảo bạn có file requirements.txt.
2. Cài đặt dependencies bằng lệnh: `pip install -r requirements.txt`.
3. Chạy script chính: `python main.py`.

## Hướng dẫn sử dụng
- Mở camera bằng cách chạy script.
- Đưa khuôn mặt của bạn vào khung hình.
- Xem kết quả hiển thị trực tiếp: Tuổi, Giới tính và Cảm xúc.
- Nhấn phím 'q' để thoát.

## Khắc phục sự cố
- Nếu camera không mở: Kiểm tra xem camera có bị khóa hoặc driver chưa cài.
- Nếu lỗi import: Cài lại dependencies với `pip install --upgrade deepface opencv-python`.
- Nếu nhận diện sai: Đảm bảo ánh sáng tốt và khoảng cách phù hợp; thử backend khác trong code (ví dụ: 'mtcnn').

## Mẹo hay
- Thử nghiệm với các backend khác nhau trong code để tăng độ chính xác.
- Thêm log debug để theo dõi kết quả phân tích.
- Dự án này dựa trên DeepFace phiên bản 0.0.93, hãy cập nhật nếu có phiên bản mới.

Cảm ơn bạn đã sử dụng dự án! Nếu có vấn đề, hãy kiểm tra log hoặc liên hệ qua issues trên GitHub. 😊 