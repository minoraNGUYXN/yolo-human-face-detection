from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras import layers, Model  # Import Keras modules

def load_embedding_model(input_shape=(160, 160, 3), embedding_dim=64):
    """Load mô hình embedding khuôn mặt."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)
    model = Model(inputs, embeddings, name='embedding_model')
    model.load_weights("models/face_embedding_model_64.h5")  # Đảm bảo đường dẫn chính xác
    return model

def get_face_embedding(face_img, model):
    """Trích xuất embedding từ ảnh khuôn mặt sử dụng mô hình đã cho."""
    try:
        resized_face = cv2.resize(face_img, (160, 160))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)
        embedding = model.predict(input_tensor)[0].tolist()
        return embedding
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        return None

def store_face_data(user_id, name, face_embedding, collection):
    """Lưu trữ dữ liệu khuôn mặt vào MongoDB."""
    try:
        if not isinstance(user_id, str):
            raise ValueError(f"user_id must be a string, got {type(user_id)}")
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")
        if not isinstance(face_embedding, list) or not all(isinstance(x, (int, float)) for x in face_embedding):
            raise ValueError(f"face_embedding must be a list of numbers, got {type(face_embedding)}")

        face_data = {
            "user_id": user_id,
            "name": name,
            "face_embedding": face_embedding,
            "created_at": datetime.now(timezone.utc),  # Use timezone-aware datetime
            "updated_at": datetime.now(timezone.utc),  # Use timezone-aware datetime
        }
        result = collection.insert_one(face_data)  # Capture the result
        print(f"Stored face data for user_id: {user_id}, inserted_id: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"Error storing face data: {e}")  # Print the full exception
        return False

def main():
    """Chức năng chính để tải ảnh và thông tin lên."""
    # Kết nối MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Thay đổi nếu cần
    db = client["face_recognition_db"]  # Thay đổi tên database nếu cần
    face_collection = db["faces"]

    # Load mô hình
    embedding_model = load_embedding_model()

    # Thông tin người dùng và ảnh (thay đổi thông tin này)
    user_id = "1"  # ID người dùng
    name = "Lê Hoàng Sơn"  # Tên người dùng
    image_path = "test_imgs/LeHoangSon.jpg"  # Đường dẫn đến ảnh khuôn mặt

    # Đọc ảnh
    face_img = cv2.imread(image_path)
    if face_img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Trích xuất embedding
    face_embedding = get_face_embedding(face_img, embedding_model)
    if face_embedding is None:
        print("Error: Could not extract face embedding.")
        return

    # Lưu vào MongoDB
    if store_face_data(user_id, name, face_embedding, face_collection):
        print("Face data uploaded successfully.")
    else:
        print("Face data upload failed.")

if __name__ == "__main__":
    main()