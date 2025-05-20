from pymongo import MongoClient
from datetime import datetime, timezone
from fastapi import HTTPException
import numpy as np

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Thay đổi nếu cần
db = client["face_recognition_db"]
face_collection = db["faces"]

def store_face_data(user_id, name, face_embedding):
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
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = face_collection.insert_one(face_data)
        print(f"Stored face data for user_id: {user_id}, inserted_id: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"Error storing face data: {e}")
        return False

def find_similar_faces(query_embedding, top_k=1):
    """Tìm kiếm các khuôn mặt tương đồng trong MongoDB."""
    try:
        query_vector = np.array(query_embedding, dtype=np.float32).tolist()
        results = face_collection.find(
            {
                "face_embedding": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": query_vector,
                        },
                        "$maxDistance": 10.0,  # Ngưỡng khoảng cách, cần điều chỉnh
                    }
                }
            },
            {"_id": 0, "name": 1}  # Projection (trường cần lấy)
        ).limit(top_k)
        return list(results)
    except Exception as e:
        print(f"Error finding similar faces: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar faces")