#!/usr/bin/env python3
from os import putenv
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2

# 1) Thiết lập biến môi trường cho ROCm (nếu cần)
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

# 2) Hàm build embedding model (dùng nếu load weights .h5 hoặc SavedModel)
def build_embedding_model(input_shape=(160,160,3), embedding_dim=128):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)

    # L2‑normalize với output_shape để tránh lỗi khi load lại
    def l2_norm(t): return tf.math.l2_normalize(t, axis=1)
    def l2_out_shape(shape): return shape

    embeddings = layers.Lambda(l2_norm, output_shape=l2_out_shape, name='l2_norm')(x)
    return Model(inputs, embeddings, name='embedding_model')

# 3) Hàm tiền xử lý ảnh
def preprocess_image(img_path, target_size=(160,160)):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size).astype(np.float32)
    img = (img / 127.5) - 1.0
    return np.expand_dims(img, axis=0)  # shape (1,H,W,C)

# 4) Parser CLI
parser = argparse.ArgumentParser(description="Benchmark face-embedding inference time")
parser.add_argument("--model", default='models/face_embedding_model_64.h5',
                    help="Path to model (.h5, SavedModel dir, or .tflite)")
parser.add_argument("--img_dir", default='dataset/vggface2/val/n000001',
                    help="Directory of images to test")
args = parser.parse_args()

# 5) Load model or interpreter
use_tflite = args.model.lower().endswith(".tflite")
if use_tflite:
    print("🔄 Loading TFLite interpreter...")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    print("🔄 Loading Keras model...")
    # Nếu là HDF5
    if args.model.lower().endswith(".h5") or args.model.lower().endswith(".keras"):
        emb_model = build_embedding_model(embedding_dim=64)
        emb_model.load_weights(args.model)
    else:
        # SavedModel directory
        emb_model = tf.keras.models.load_model(args.model)
    # Warm‑up Keras model and wrap
    sample_img = next(iter(os.listdir(args.img_dir)))
    sample = preprocess_image(os.path.join(args.img_dir, sample_img))
    _ = emb_model(sample, training=False)
    infer_fn = tf.function(lambda x: emb_model(x, training=False),
                           input_signature=[tf.TensorSpec([1,160,160,3], tf.float32)])
    _ = infer_fn(sample)

# 6) Tập hợp danh sách ảnh
img_paths = [
    os.path.join(args.img_dir, fn)
    for fn in sorted(os.listdir(args.img_dir))
    if fn.lower().endswith(('.jpg','.png','.jpeg'))
]
n_images = len(img_paths)
if n_images == 0:
    raise ValueError(f"No images found in {args.img_dir}")

# 7) Đo thời gian inference
total_time = 0.0
for p in img_paths:
    img = preprocess_image(p)
    start = time.perf_counter()

    if use_tflite:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img.astype(input_details[0]['dtype']))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    else:
        _ = infer_fn(img)

    # Nếu GPU, đồng bộ
    try:
        tf.experimental.sync_devices()
    except:
        pass

    end = time.perf_counter()
    total_time += (end - start)

print(f"\n📊 Processed {n_images} images in {total_time:.6f} seconds")
print(f"⚡ Average per image: {total_time / n_images:.6f} seconds")