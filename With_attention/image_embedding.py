import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
import logging
from typing import List, Dict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def call(self, inputs):
        query, key, value = inputs
        scores = tf.matmul(query, key, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, value)
        return output

class ImageEmbeddingModel:
    def __init__(self):
        self.model = self._create_model()
        # self.model.eval()

    def _create_model(self):
        inputs = layers.Input(shape=(224, 224, 3))

        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Flatten and add attention
        x = layers.Flatten()(x)
        query = layers.Dense(64)(x)
        key = layers.Dense(64)(x)
        value = layers.Dense(64)(x)

        x = AttentionLayer()([query, key, value])

        # Final layer for embedding
        x = layers.Dense(128, activation='relu')(x)
        model = Model(inputs, x)

        return model

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))  # Resize to match model input
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            embedding = self.model.predict(image)  # Use predict instead of forward
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None