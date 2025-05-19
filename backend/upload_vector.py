import redis
import numpy as np
from datetime import datetime

# Kết nối Redis
r = redis.Redis(host='localhost', port=6379)

# Tên index bạn muốn tạo
index_name = "test_index"

# Kiểm tra nếu index chưa tồn tại thì mới tạo
if index_name.encode() not in r.execute_command("FT._LIST"):
    try:
        r.execute_command(
            "FT.CREATE", index_name,
            "ON", "HASH",
            "PREFIX", "1", "doc:",
            "SCHEMA",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", "128", "DISTANCE_METRIC", "COSINE",
            "timestamp", "TEXT",
            "name", "TEXT"
        )
        print(f"Đã tạo index {index_name}")
    except redis.exceptions.ResponseError as e:
        print("Lỗi khi tạo index:", e)
else:
    print(f"Index '{index_name}' đã tồn tại.")

# Tạo và lưu vector
for i in range(1, 541):
    # Tạo vector ngẫu nhiên
    vector = np.random.rand(128).astype(np.float32)
    # Lưu vector vào Redis
    r.hset(f"doc:{i}", mapping={
        "embedding": vector.tobytes(),
        "timestamp": datetime.now().isoformat(),
        "name": f"person_{i}"
    })
    print(f"Đã lưu vector cho doc:{i}")