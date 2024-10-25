# Advanced Image Similarity Search System

This repository contains two implementations of an image similarity search system using different deep learning approaches: ResNet50 and an attention-based model. The system allows you to find similar images from a database based on a query image.

## Implementations



### 1. Attention-based Implementation
- Uses a custom CNN architecture with attention mechanism
- Implemented using TensorFlow/Keras
- Features a custom attention layer for improved feature focus
- Lightweight architecture with configurable parameters

### 2. ResNet50-based Implementation
- Uses pre-trained ResNet50 model for feature extraction
- Leverages transfer learning for robust image embeddings
- Implemented using PyTorch framework
- Features automatic GPU utilization when available

## Project Structure

```
├── image_embedding.py     # Image embedding model implementations
├── image_search_system.py # Main search system implementation
├── utils.py              # Utility functions for visualization
└── main.py              # Example usage and demo script
```

## Dependencies

- Python 3.6+
- PyTorch (for ResNet50 version)
- TensorFlow (for Attention version)
- PIL (Python Imaging Library)
- NumPy
- FAISS
- Matplotlib
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ayoub-Elkhaiari/Image_Search_with_Attention_mecanism.git
cd Image_Search_with_Attention_mecanism
```

2. Install required packages:
```bash
pip install torch torchvision numpy pillow faiss-cpu matplotlib tqdm tensorflow
```

## Usage

### Basic Usage

```python
from image_search_system import ImageSearchSystem
from utils import visualize_results

# Initialize the search system
search_system = ImageSearchSystem()

# Add images to the system
image_directory = "path/to/image/directory"
search_system.add_images(image_directory)

# Perform a search
query_image = "path/to/query/image.jpg"
results = search_system.search(query_image, k=5)

# Visualize results
visualize_results(query_image, results)
```

### Choosing an Implementation

#### ResNet50 Version (PyTorch)
- Better for transfer learning scenarios
- More suitable for general-purpose image similarity
- Requires more computational resources
- Located in the `with_ResNet50_Embedding` directory

#### Attention-based Version (TensorFlow)
- More lightweight and customizable
- Features attention mechanism for focused feature extraction
- Better for specific domain applications
- Located in the `With_attention` directory

## Key Components

### ImageEmbeddingModel
- Handles image preprocessing
- Generates embeddings using neural networks
- Implements different architectures based on version

### ImageSearchSystem
- Manages the image database
- Builds and maintains FAISS index
- Handles similarity search operations

### Visualization
- Provides tools for result visualization
- Displays query image and similar matches
- Shows similarity scores for each match

## Features

- Fast similarity search using FAISS indexing
- Batch processing for efficient image handling
- Visualization tools for search results
- Configurable number of search results (k)
- Support for various image formats (PNG, JPG, JPEG, GIF, BMP)
- Error handling and logging

## Performance Considerations

- ResNet50 version:
  - Higher accuracy for general-purpose image similarity
  - Requires more memory and computational power
  - Benefits from GPU acceleration

- Attention-based version:
  - Faster training and inference
  - Lower memory footprint
  - More suitable for custom domain adaptation

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, and suggest features.

## Directory Structure for Images

```
├── data
│   ├── obj_decoys/      # Database images
│   └── img_requetes/    # Query images directory
```

## Logging

The system includes comprehensive logging:
- Initialization status
- Processing progress
- Error handling
- Success confirmations

## Results
   With Attention:

  ![Screenshot 2024-10-25 164634](https://github.com/user-attachments/assets/604848cb-1429-4e6b-a819-604bcbf51070)


  Without Attention:

  ![Screenshot 2024-10-25 152649](https://github.com/user-attachments/assets/6b56b854-199c-4ec6-9dc3-3baf0cec0c91)
   

From the results, we note that scores are very high when it comes to attention. It turns out that Attention mechanism tried to attach attention weight to the important features in this case the buildings (castles in images) and retrieve the similar ones from the vectorstore.

---

