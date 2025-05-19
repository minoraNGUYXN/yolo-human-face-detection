from os import putenv
import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# 1) Thiết lập biến môi trường cho ROCm (nếu cần)
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

# 2) Định nghĩa model embedding (bao gồm cả Lambda với output_shape)
def build_embedding_model(input_shape=(160,160,3), embedding_dim=128):
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

    return Model(inputs, embeddings, name='embedding_model')

# 3) Build và load weights
emb_model = build_embedding_model(embedding_dim=64)
emb_model.load_weights("models/face_embedding_model_64.h5")
print("✔️ Loaded weights into embedding model.")

# 4) Export model sang SavedModel bằng Model.export()
export_dir = "models/saved_face_embedding"
if os.path.exists(export_dir):
    tf.io.gfile.rmtree(export_dir)

emb_model.export(export_dir)
print(f"✔️ Model exported to SavedModel at '{export_dir}'")

# 5) Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()
tflite_path = "models/face_embedding_model_64_fp16.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"✔️ TFLite model saved to '{tflite_path}'")