from os import putenv

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0/")

import numpy as np
import cv2
from .models import load_models, get_emotion_model_details, get_action_model_details, get_face_embedding_model_details
from .config import EMOTION_LABELS, ACTION_LABELS

class Detector:
    """Xử lý nhận diện người, khuôn mặt và cảm xúc / Detection handler"""
    def __init__(self):
        """
        Khởi tạo Detector bằng cách tải các mô hình.
        Initializes the Detector by loading the models.
        """
        # Tải các mô hình / Load models
        self.person_model, self.face_model, self.emotion_interpreter, self.action_interpreter, self.face_embedding_interpreter = load_models()
        
        # Lấy thông tin chi tiết về mô hình cảm xúc / Get emotion model details
        self.emotion_input_details, self.emotion_output_details = get_emotion_model_details(self.emotion_interpreter)
        
        # Lấy thông tin chi tiết về mô hình hành vi / Get action model details
        self.action_input_details, self.action_output_details = get_action_model_details(self.action_interpreter)
        
        # Lấy kích thước đầu vào của mô hình cảm xúc / Get emotion model input size
        self.emotion_input_shape = self.emotion_input_details[0]['shape']
        self.emotion_height = self.emotion_input_shape[1]
        self.emotion_width = self.emotion_input_shape[2]
        
        # Lấy kích thước đầu vào của mô hình hành vi / Get action model input size
        self.action_input_shape = self.action_input_details[0]['shape']
        self.action_height = self.action_input_shape[1]
        self.action_width = self.action_input_shape[2]

        # Lấy thông tin chi tiết của mô hình embedding
        self.face_embedding_input_details, self.face_embedding_output_details = get_face_embedding_model_details(self.face_embedding_interpreter) # Thêm dòng này
        self.face_embedding_input_shape = self.face_embedding_input_details[0]['shape'] # Thêm dòng này
        self.embedding_size = self.face_embedding_input_shape[-1] # kích thước của vector embedding

        # Lưu trữ mô hình embedding khuôn mặt
        self.emb_model = self.face_embedding_interpreter
    
    def process_frame(self, frame):
        """
        Xử lý khung hình và trả về kết quả nhận diện.
        Processes a frame and returns the detection results.

        Args:
            frame (np.ndarray): Khung hình đầu vào (ảnh). Input frame (image).

        Returns:
            tuple: Một tuple chứa:
                   - person_count (int): Số lượng người được nhận diện. Number of detected persons.
                   - face_count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                   - person_boxes (list): Danh sách các khung người với thông tin hành vi. List of person bounding boxes with action info.
                   - face_boxes (list): Danh sách các khung khuôn mặt với thông tin cảm xúc. List of face bounding boxes with emotion info.
        """
        try:
            # Nhận diện người / Detect persons
            person_results = self.person_model(frame, classes=[0], conf=0.3, iou=0.45, imgsz=640, half=True, verbose=False)
            
            person_count = 0
            face_count = 0
            face_boxes = []  # Danh sách khung khuôn mặt / Face boxes list
            person_boxes = []  # Danh sách khung người có thông tin hành vi / Person boxes list with action info
            
            if person_results:
                for person in person_results:
                    # Lấy tọa độ khung người / Get person boxes
                    boxes = person.boxes.xyxy.cpu().numpy() #boxes is a numpy array with shape (n,4)
                    confs = person.boxes.conf.cpu().numpy() #confidence for person box
                    person_count += len(boxes)

                    # Nhận diện hành vi và lưu thông tin / Detect actions and save info
                    for i, (box, conf) in enumerate(zip(boxes, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Cắt vùng ảnh người để nhận diện hành vi / Get person ROI for action detection
                        if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                            person_roi = frame[y1:y2, x1:x2]
                            
                            # Chỉ xử lý nếu vùng ảnh người hợp lệ / Only process if person ROI is valid
                            if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                                action = self._detect_action(person_roi)
                            else:
                                action = "Không xác định"  # Không thể xác định hành vi / Unknown action
                        else:
                            action = "Không xác định"
                        
                        # Lưu thông tin khung người kèm hành vi / Save person box with action info
                        person_boxes.append(((x1, y1, x2, y2), float(conf), action))
                    
                    # Chuẩn bị vùng quan tâm (ROI) cho khuôn mặt / Prepare ROIs for face detection
                    valid_rois, valid_indices = self._prepare_rois(frame, boxes)
                    # Nhận diện khuôn mặt và cảm xúc trong ROI / Detect faces and emotions in ROIs
                    face_data = self._detect_faces_and_emotions(frame, valid_rois, valid_indices, boxes)
                    face_count += face_data["count"]
                    face_boxes.extend(face_data["boxes"])
            
            return person_count, face_count, person_boxes, face_boxes
            
        except Exception as e:
            # Xử lý lỗi tại đây để tránh gây đơ ứng dụng / Handle errors to avoid freezing
            print(f"Error in process_frame: {e}")
            # Trả về giá trị mặc định an toàn / Return safe default values
            return 0, 0, [], []
    
    def _prepare_rois(self, frame, boxes):
        """
        Chuẩn bị các vùng ảnh hợp lệ từ khung người.
        Extracts valid image regions from person bounding boxes.
        """
        valid_rois = []
        valid_indices = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if (x2 > x1) and (y2 > y1) and (x1 >= 0) and (y1 >= 0):
                valid_rois.append(frame[y1:y2, x1:x2])  # Cắt vùng ảnh / Crop region
                valid_indices.append(i)
        return valid_rois, valid_indices
    
    def _detect_faces_and_emotions(self, original_frame, rois, indices, boxes):
        """
        Nhận diện khuôn mặt, cảm xúc và tính toán tọa độ toàn cục.
        Detects faces and emotions and calculates global coordinates.

        Args:
            original_frame (np.ndarray): Khung hình gốc. Original frame.
            rois (list): Danh sách các vùng ảnh (ROI). List of image regions (ROIs).
            indices (list): Danh sách các chỉ số tương ứng với các khung người. List of indices corresponding to person boxes.
            boxes (np.ndarray): Mảng các khung người. Array of person bounding boxes.

        Returns:
            dict: Một dictionary chứa:
                  - count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                  - boxes (list): Danh sách các khung khuôn mặt với thông tin cảm xúc. List of face bounding boxes with emotion info.
        """
        face_boxes, face_count = [], 0
        for roi_idx, roi in enumerate(rois):
            try:
                result = self.face_model(roi, conf=0.3, iou=0.45, imgsz=160, half=True, verbose=False)[0]
                if not result.boxes:
                    continue
                # Chỉ lấy khuôn mặt có confidence cao nhất / Select the single best face
                best = max(result.boxes, key=lambda b: float(b.conf.cpu().numpy()))
                fx1,fy1,fx2,fy2 = map(int, best.xyxy[0].cpu().numpy())
                conf = float(best.conf[0].cpu().numpy())
                px1,py1 = int(boxes[indices[roi_idx]][0]), int(boxes[indices[roi_idx]][1])
                # Crop face region
                face_roi = roi[fy1:fy2, fx1:fx2] if fx2>fx1 and fy2>fy1 else None
                if face_roi is None or face_roi.size==0:
                    continue
                emotion = self._detect_emotion(face_roi)
                embedding = self._get_face_embedding(face_roi)
                global_box = (px1+fx1, py1+fy1, px1+fx2, py1+fy2)
                face_boxes.append((global_box, conf, emotion, embedding))
                face_count += 1
            except Exception as e:
                print(f"Error in face ROI {roi_idx}: {e}")
                continue
        return {'count': face_count, 'boxes': face_boxes}
    
    def _detect_emotion(self, face_img):
        """
        Phát hiện cảm xúc từ ảnh khuôn mặt sử dụng mô hình TFLite
        Detect emotion from a face image using the TFLite model
        
        Args:
            face_img (np.ndarray): Vùng ảnh khuôn mặt / Face image region
            
        Returns:
            str: Nhãn cảm xúc dự đoán / Predicted emotion label
        """
        try:
            # Tiền xử lý ảnh khuôn mặt cho mô hình cảm xúc / Preprocess face image for emotion model
            
            # Kiểm tra xem cần chuyển sang ảnh xám không / Check if grayscale is needed
            if self.emotion_input_shape[-1] == 1:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Thay đổi kích thước ảnh theo yêu cầu đầu vào / Resize to expected input dimensions
            resized_face = cv2.resize(face_img, (self.emotion_width, self.emotion_height))
            
            # Kiểm tra kiểu dữ liệu đầu vào từ chi tiết / Check input type from details
            input_dtype = self.emotion_input_details[0]['dtype']
            
            # Xử lý tùy thuộc vào kiểu dữ liệu đầu vào / Process based on input data type
            if input_dtype == np.float32:
                # Chuẩn hóa giá trị pixel cho mô hình float / Normalize pixel values for float models
                normalized_face = resized_face.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                # Đối với mô hình lượng tử hóa, giữ nguyên dạng uint8 / For quantized models, keep as uint8
                normalized_face = resized_face.astype(np.uint8)
            else:
                # Mặc định chuẩn hóa kiểu float32 / Default to float32 normalization
                normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Thay đổi hình dạng để phù hợp với đầu vào / Reshape to match input tensor shape
            if self.emotion_input_shape[-1] == 1:
                input_tensor = normalized_face.reshape(1, self.emotion_height, self.emotion_width, 1)
            else:
                input_tensor = normalized_face.reshape(1, self.emotion_height, self.emotion_width, 3)
            
            # Đặt tensor đầu vào / Set input tensor
            self.emotion_interpreter.set_tensor(self.emotion_input_details[0]['index'], input_tensor)
            
            # Chạy suy luận / Run inference
            self.emotion_interpreter.invoke()
            
            # Lấy tensor đầu ra / Get output tensor
            output_tensor = self.emotion_interpreter.get_tensor(self.emotion_output_details[0]['index'])
            
            # Lấy cảm xúc dự đoán / Get predicted emotion
            emotion_idx = np.argmax(output_tensor)
            
            # Đảm bảo chỉ số nằm trong giới hạn của danh sách nhãn / Ensure index is within range of labels
            if 0 <= emotion_idx < len(EMOTION_LABELS):
                emotion_label = EMOTION_LABELS[emotion_idx]
            else:
                emotion_label = "Không xác định"
            
            return emotion_label
            
        except Exception as e:
            print(f"Lỗi khi nhận diện cảm xúc: {e}")
            return "Không xác định"

    def _detect_action(self, person_img):
        """
        Phát hiện hành vi từ ảnh người sử dụng mô hình TFLite
        Detect action from a person image using the TFLite model
        
        Args:
            person_img (np.ndarray): Vùng ảnh người / Person image region
            
        Returns:
            str: Nhãn hành vi dự đoán / Predicted action label
        """
        try:
            # Thay đổi kích thước ảnh theo yêu cầu đầu vào / Resize to expected input dimensions
            resized_person = cv2.resize(person_img, (self.action_width, self.action_height))
            
            # Kiểm tra số kênh màu đầu vào / Check input channels
            expected_channels = self.action_input_shape[-1]
            
            # Chuyển đổi sang ảnh xám nếu cần / Convert to grayscale if needed
            if expected_channels == 1:
                processed_person = cv2.cvtColor(resized_person, cv2.COLOR_BGR2GRAY)
            else:
                processed_person = resized_person
            
            # Kiểm tra kiểu dữ liệu đầu vào từ chi tiết / Check input type from details
            input_dtype = self.action_input_details[0]['dtype']
            
            # Xử lý tùy thuộc vào kiểu dữ liệu đầu vào / Process based on input data type
            if input_dtype == np.float32:
                # Chuẩn hóa giá trị pixel cho mô hình float / Normalize pixel values for float models
                normalized_person = processed_person.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                # Đối với mô hình lượng tử hóa, giữ nguyên dạng uint8 / For quantized models, keep as uint8
                normalized_person = processed_person.astype(np.uint8)
            else:
                # Mặc định chuẩn hóa kiểu float32 / Default to float32 normalization
                normalized_person = processed_person.astype(np.float32) / 255.0
            
            # Thay đổi hình dạng để phù hợp với đầu vào / Reshape to match input tensor shape
            if expected_channels == 1:
                input_tensor = normalized_person.reshape(1, self.action_height, self.action_width, 1)
            else:
                input_tensor = normalized_person.reshape(1, self.action_height, self.action_width, expected_channels)
            
            # Đặt tensor đầu vào / Set input tensor
            self.action_interpreter.set_tensor(self.action_input_details[0]['index'], input_tensor)
            
            # Chạy suy luận / Run inference
            self.action_interpreter.invoke()
            
            # Lấy tensor đầu ra / Get output tensor
            output_tensor = self.action_interpreter.get_tensor(self.action_output_details[0]['index'])
            
            # Lấy hành vi dự đoán / Get predicted action
            action_idx = np.argmax(output_tensor)
            
            # Đảm bảo chỉ số nằm trong giới hạn của danh sách nhãn / Ensure index is within range of labels
            if 0 <= action_idx < len(ACTION_LABELS):
                action_label = ACTION_LABELS[action_idx]
            else:
                action_label = "Không xác định"
            
            return action_label
            
        except Exception as e:
            print(f"Lỗi khi nhận diện hành vi: {e}")
            return "Không xác định"

    def _get_face_embedding(self, face_img):
        try:
            resized_face = cv2.resize(face_img, (self.face_embedding_input_shape[1], self.face_embedding_input_shape[2])) # Sử dụng kích thước từ mô hình
            
            # Kiểm tra kiểu dữ liệu đầu vào của mô hình embedding
            input_dtype = self.face_embedding_input_details[0]['dtype']
            if input_dtype == np.float32:
                normalized_face = resized_face.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                normalized_face = resized_face.astype(np.uint8)
            else:
                normalized_face = resized_face.astype(np.float32) / 255.0
                
            input_tensor = np.expand_dims(normalized_face, axis=0)
            
            self.emb_model.set_tensor(self.face_embedding_input_details[0]['index'], input_tensor)
            self.emb_model.invoke()
            embedding = self.emb_model.get_tensor(self.face_embedding_output_details[0]['index'])[0].tolist()
            return embedding
        except Exception as e:
            print(f"Error getting face embedding: {e}")
            return None