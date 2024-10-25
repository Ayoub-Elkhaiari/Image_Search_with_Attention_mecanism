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

from utils import visualize_results
from image_search_system import ImageSearchSystem




def main():
    # Initialize the search system
    search_system = ImageSearchSystem()
    
    # Add images to the system
    image_directory = "With_attention\data\obj_decoys"  # Use forward slashes for paths
    search_system.add_images(image_directory)
    
    # Perform a search
    query_image = "With_attention\data\img_requetes\ImageRequete.jpg"
    results = search_system.search(query_image, k=5)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Image: {result['filename']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Path: {result['path']}")
    
    # Visualize results
    fig = visualize_results(query_image, results)
    plt.show()

if __name__ == "__main__":
    main()
