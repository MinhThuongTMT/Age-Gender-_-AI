import cv2
from deepface import DeepFace

def main():
    # Mở camera (0 là camera mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            # Phân tích tuổi, giới tính và cảm xúc từ frame
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],  # Các thuộc tính cần phân tích
                enforce_detection=False,              # Không bắt buộc phát hiện khuôn mặt
                detector_backend='mtcnn'              # Sử dụng MTCNN làm detector
            )

            # Kiểm tra kết quả trả về từ DeepFace
            if isinstance(results, list) and results:
                result = results[0]  # Lấy kết quả đầu tiên (nếu có nhiều khuôn mặt)

                # Lấy tuổi
                age = result['age']

                # Xử lý giới tính
                gender_raw = result.get('gender')
                print(f"Debug: Giá trị raw của giới tính: {gender_raw}")
                if isinstance(gender_raw, str):
                    # Nếu gender_raw là chuỗi
                    gender_lower = gender_raw.lower()
                    gender_mapped = 'Man' if gender_lower == 'man' else 'Nữ' if gender_lower == 'woman' else 'Unknown'
                elif isinstance(gender_raw, dict):
                    # Nếu gender_raw là từ điển chứa xác suất
                    if gender_raw and all(isinstance(v, (int, float)) for v in gender_raw.values()):
                        dominant_gender = max(gender_raw, key=lambda k: gender_raw[k])
                        dominant_gender_lower = dominant_gender.lower()
                        gender_mapped = 'Man' if dominant_gender_lower == 'man' else 'Nữ' if dominant_gender_lower == 'woman' else 'Unknown'
                    else:
                        gender_mapped = 'Unknown'
                else:
                    gender_mapped = 'Unknown'

                # Xử lý cảm xúc
                emotion_raw = result.get('emotion', {})
                if isinstance(emotion_raw, dict) and emotion_raw:
                    dominant_emotion = max(emotion_raw, key=lambda k: emotion_raw[k])
                    emotion_text = f"{dominant_emotion} ({emotion_raw[dominant_emotion]:.2f}%)"
                else:
                    emotion_text = 'Unknown'

                # In thông tin debug ra console
                print(f"Debug: Phát hiện tuổi {age}, giới tính {gender_mapped}, cảm xúc {emotion_text}")

                # Vẽ kết quả lên frame
                cv2.putText(frame, f"Age: {age}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {gender_mapped}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion_text}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Debug: Không phát hiện khuôn mặt")
        except Exception as e:
            print(f"Error in analysis: {e}")

        # Hiển thị frame với kết quả
        cv2.imshow('Age, Gender, and Emotion Detection', frame)

        # Thoát vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()