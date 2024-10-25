from PIL import Image
from matplotlib import pyplot as plt 
from typing import List, Dict

def visualize_results(query_image_path: str, results: List[Dict], figsize=(15, 8)):
    """Visualize query image and search results in a grid."""
    n_results = len(results)
    fig = plt.figure(figsize=figsize)
    
    # Create a grid with 2 rows: query image on top, results on bottom
    gs = plt.GridSpec(2, max(n_results, 1))
    
    # Plot query image
    ax = fig.add_subplot(gs[0, :])
    query_img = Image.open(query_image_path)
    ax.imshow(query_img)
    ax.set_title("Query Image", fontsize=12, pad=10)
    ax.axis('off')
    
    # Plot results
    for idx, result in enumerate(results):
        ax = fig.add_subplot(gs[1, idx])
        img = Image.open(result['path'])
        ax.imshow(img)
        ax.set_title(f"Score: {result['similarity_score']:.3f}\n{result['filename']}", 
                    fontsize=10, pad=5)
        ax.axis('off')
    
    plt.tight_layout()
    return fig 
