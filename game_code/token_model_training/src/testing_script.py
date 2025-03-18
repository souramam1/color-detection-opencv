import os
from token_labelling import TokenLabeller

def test_folder_structure():
    color = "magenta"
    output_folder = "Image_Processing_Improvements/token-detection-training/labelled_data"
    
    # Create an instance of TokenLabeller
    labeller = TokenLabeller(color, output_folder)
    
    # Check if the folder structure is created correctly
    color_folder = os.path.join(output_folder, f"{color}_tokens")
    if os.path.exists(color_folder):
        print(f"Folder structure created correctly: {color_folder}")
    else:
        print(f"Failed to create folder structure: {color_folder}")

if __name__ == "__main__":
    test_folder_structure()