from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

import time
import numpy as np
import cv2
from pymongo import MongoClient
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras import layers, Model
import io

def load_embedding_model(input_shape=(160, 160, 3), embedding_dim=64):
    """
    Tải mô hình embedding khuôn mặt.

    Args:
        input_shape (tuple, optional): Kích thước ảnh đầu vào. Mặc định là (160, 160, 3).
        embedding_dim (int, optional): Kích thước của vector embedding. Mặc định là 128.

    Returns:
        tensorflow.keras.Model: Mô hình embedding đã được tải.
    """
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
    """
    Trích xuất embedding từ ảnh khuôn mặt bằng mô hình đã cho.

    Args:
        face_img (numpy.ndarray): Ảnh khuôn mặt đầu vào.
        model (tensorflow.keras.Model): Mô hình embedding đã tải.

    Returns:
        list: Vector embedding của khuôn mặt, hoặc None nếu có lỗi.
    """
    try:
        resized_face = cv2.resize(face_img, (160, 160))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)
        embedding = model.predict(input_tensor)[0].tolist()
        return embedding
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding: {e}")
        return None

def measure_latency_and_query(image_path, top_k=5, num_runs=100, mongo_uri="mongodb://localhost:27017/"):
    """
    Đo độ trễ khi trích xuất embedding từ ảnh và truy vấn các khuôn mặt tương tự từ MongoDB.

    Args:
        image_path (str): Đường dẫn đến ảnh khuôn mặt đầu vào.
        top_k (int, optional): Số lượng kết quả tương tự trả về. Mặc định là 5.
        num_runs (int, optional): Số lần chạy truy vấn để tính trung bình. Mặc định là 100.
        mongo_uri (str, optional): Chuỗi kết nối MongoDB. Mặc định là "mongodb://localhost:27017/".

    Returns:
        float: Độ trễ trung bình của các truy vấn tính bằng mili giây, hoặc None nếu có lỗi.
    """
    try:
        # Kết nối MongoDB
        client = MongoClient(mongo_uri)
        db = client["face_recognition_db"]
        face_collection = db["faces"]

        # Tạo index 2dsphere nếu nó chưa tồn tại
        face_collection.create_index([("face_embedding", "2dsphere")])

        # Load mô hình embedding
        embedding_model = load_embedding_model()

        # Đọc ảnh từ đường dẫn
        face_img = cv2.imread(image_path)
        if face_img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

        # Trích xuất embedding từ ảnh
        query_embedding = get_face_embedding(face_img, embedding_model)
        if query_embedding is None:
            raise ValueError("Không thể trích xuất embedding từ ảnh.")

        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            results = face_collection.find(
                {
                    "face_embedding": {
                        "$near": {
                            "$geometry": {
                                "type": "Point",
                                "coordinates": query_embedding,
                            },
                            "$maxDistance": 0.5,  # Ngưỡng khoảng cách, cần điều chỉnh
                        },
                    },
                },
                {"_id": 0, "name": 1},  # Chỉ lấy trường "name"
            ).limit(top_k)
            list(results)  # Đảm bảo query được thực thi và fetch kết quả
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # Đổi sang mili giây

        average_latency = total_time / num_runs
        print(f"Độ trễ trung bình khi trích xuất embedding và truy vấn MongoDB ({num_runs} lần chạy): {average_latency:.2f} ms")
        return average_latency

    except Exception as e:
        print(f"Lỗi: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()  # Đóng kết nối MongoDB

if __name__ == "__main__":
    # Thay đổi đường dẫn ảnh thành đường dẫn ảnh của bạn
    image_path = "test_imgs/LeHoangSon.jpg"  #  Đường dẫn ảnh
    mongo_uri = "mongodb://localhost:27017/" # Chuỗi kết nối MongoDB
    # Đo độ trễ và truy vấn
    latency = measure_latency_and_query(image_path, mongo_uri=mongo_uri)
    if latency is not None:
        print(f"Độ trễ trung bình: {latency:.2f} ms")