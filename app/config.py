# Đường dẫn mô hình / Model paths
PERSON_MODEL_PATH = './models/yolov8n-person-lw.pt'
FACE_MODEL_PATH = './models/yolov8p-face.pt'
EMOTION_MODEL_PATH = './models/emotion_model_2.tflite'

# Thiết lập hiển thị / Visualization settings
PERSON_COLOR = (0, 0, 255)   # Màu đỏ / Red color
FACE_COLOR = (0, 255, 0)     # Màu xanh lá / Green color
EMOTION_COLOR = (255, 165, 0)  # Màu cam / Orange color for emotion
TEXT_SCALE = 1             # Tỷ lệ chữ / Text scale
TEXT_THICKNESS = 2           # Độ dày chữ / Text thickness

# Danh sách cảm xúc / Emotion labels
EMOTION_LABELS = ['Giận dữ', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Bình thường']