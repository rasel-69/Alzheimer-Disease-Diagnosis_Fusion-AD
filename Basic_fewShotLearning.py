import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import random
import os
from pathlib import Path
from PIL import Image

# ---------- CONFIG ----------
IMG_SIZE = (224, 224)
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
EPOCHS = 60

train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"

# ---------- UTILITIES ----------
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return img

def load_dataset(data_dir):
    data = {}
    class_names = sorted(os.listdir(data_dir))
    for cls in class_names:
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            data[cls] = images
    return data

# ---------- EPISODE CREATION ----------
def create_episode(data, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    selected_classes = random.sample(list(data.keys()), n_way)
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

    for label, cls in enumerate(selected_classes):
        selected_images = random.sample(data[cls], k_shot + q_query)
        support_set += [preprocess_image(img) for img in selected_images[:k_shot]]
        query_set += [preprocess_image(img) for img in selected_images[k_shot:]]
        support_labels += [label] * k_shot
        query_labels += [label] * q_query

    return (np.array(support_set), np.array(support_labels),
            np.array(query_set), np.array(query_labels))

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
import tensorflow as tf

IMG_SIZE = (224, 224)

# ---- CBAM Block ----
def cbam_block(inputs, ratio=8):
    channel = inputs.shape[-1]

    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)

    shared_dense = tf.keras.Sequential([
        layers.Dense(channel // ratio, activation='relu'),
        layers.Dense(channel)
    ])

    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)
    channel_att = layers.Add()([avg_out, max_out])
    channel_att = layers.Activation('sigmoid')(channel_att)
    channel_att = layers.Reshape((1, 1, channel))(channel_att)
    x = layers.Multiply()([inputs, channel_att])

    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial_att = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_att)
    x = layers.Multiply()([x, spatial_att])
    return x

# ---- ENHANCED RESNET50 ENCODER ----
def build_encoder():
    base_model = ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Optional: freeze backbone

    inputs = base_model.input
    x = base_model.output

    # CBAM attention on ResNet50 output
    x = cbam_block(x)

    # Additional convolutional refinements
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global feature pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # Regularization
    encoder = Model(inputs=inputs, outputs=x, name="resnet50_cbam_encoder")
    return encoder


# ---------- PROTOTYPICAL NETWORK HEAD ----------
def compute_prototypes(support_embeddings, support_labels, n_way):
    prototypes = []
    for i in range(n_way):
        cls_embeddings = tf.boolean_mask(support_embeddings, tf.equal(support_labels, i))
        prototypes.append(tf.reduce_mean(cls_embeddings, axis=0))
    return tf.stack(prototypes)

def euclidean_distance(a, b):
    return tf.norm(tf.expand_dims(a, 1) - tf.expand_dims(b, 0), axis=-1)

# ---------- TRAINING LOOP ----------
train_data = load_dataset(train_dir)
val_data = load_dataset(val_dir)
encoder = build_encoder()

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(support_images, support_labels, query_images, query_labels):
    with tf.GradientTape() as tape:
        support_embeddings = encoder(support_images, training=True)
        query_embeddings = encoder(query_images, training=True)

        prototypes = compute_prototypes(support_embeddings, support_labels, N_WAY)
        distances = euclidean_distance(query_embeddings, prototypes)
        logits = -distances

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=query_labels, logits=logits)
        loss = tf.reduce_mean(loss)

        # Compute accuracy
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_labels), tf.float32))

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    return loss, accuracy


for epoch in range(EPOCHS):
    support_imgs, support_lbls, query_imgs, query_lbls = create_episode(train_data)
    loss, accuracy = train_step(
        tf.convert_to_tensor(support_imgs, dtype=tf.float32),
        tf.convert_to_tensor(support_lbls, dtype=tf.int32),
        tf.convert_to_tensor(query_imgs, dtype=tf.float32),
        tf.convert_to_tensor(query_lbls, dtype=tf.int32),
    )
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
