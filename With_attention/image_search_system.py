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
from image_embedding import ImageEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ImageSearchSystem:
    def __init__(self):
        self.embedding_model = ImageEmbeddingModel()
        self.image_paths: List[str] = []
        self.embeddings = None
        self.index = None
        self.image_to_metadata: Dict[str, dict] = {}

    def add_images(self, image_directory: str, batch_size: int = 32):
        """Process all images in a directory and build the FAISS index."""
        logger.info("Starting image processing...")

        # Get all image paths
        image_files = [
            os.path.join(image_directory, f) for f in os.listdir(image_directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]

        embeddings_list = []
        valid_paths = []

        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i + batch_size]

            for image_path in batch_files:
                embedding = self.embedding_model.get_embedding(image_path)

                if embedding is not None:
                    embeddings_list.append(embedding)
                    valid_paths.append(image_path)

                    # Store metadata
                    self.image_to_metadata[image_path] = {
                        'filename': os.path.basename(image_path),
                        'path': image_path,
                    }

        # Convert list of embeddings to numpy array
        self.embeddings = np.vstack(embeddings_list)
        self.image_paths = valid_paths

        # Build FAISS index
        self._build_faiss_index()

        logger.info(f"Processed {len(valid_paths)} images successfully")

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity

        # Normalize vectors to use inner product as cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        logger.info("FAISS index built successfully")

    def search(self, query_image_path: str, k: int = 5) -> List[Dict]:
        """Search for similar images given a query image."""
        query_embedding = self.embedding_model.get_embedding(query_image_path)

        if query_embedding is None:
            raise ValueError("Could not process query image")

        # Reshape and normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Perform search
        distances, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths):
                image_path = self.image_paths[idx]
                result = {
                    **self.image_to_metadata[image_path],
                    'similarity_score': float(distance)
                }
                results.append(result)

        return results