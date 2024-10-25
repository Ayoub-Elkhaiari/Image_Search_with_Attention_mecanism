import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import faiss
from tqdm import tqdm
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from typing import List, Dict
import logging
from matplotlib import pyplot as plt 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbeddingModel:
    def __init__(self):
        # Load pre-trained ResNet model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(image)

            # Flatten and convert to numpy array
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None


