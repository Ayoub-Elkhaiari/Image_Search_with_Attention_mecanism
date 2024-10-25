import os
from tqdm import tqdm
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from matplotlib import pyplot as plt 
from image_search_system import ImageSearchSystem
from utils import visualize_results

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def main():
    # Initialize the search system
    search_system = ImageSearchSystem()
    
    # Add images to the system
    image_directory = "data\obj_decoys"
    search_system.add_images(image_directory)
    
    # Perform a search
    query_image = "data\img_requetes\ImageRequete2.jpg"
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

