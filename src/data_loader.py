import os
import xml.etree.ElementTree as ET
import pandas as pd
from glob import glob
import base64
from src.config import Config

class DataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Config.DATA_DIR
        self.data_dir = data_dir
        self.annotations_dir = Config.ANNOTATIONS_DIR
        self.images_dir = Config.RAW_IMAGES_DIR 

    def load_annotations(self):
        """Parses all XML files in the annotations directory."""
        xml_files = glob(os.path.join(self.annotations_dir, "*.xml"))
        data = []

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            dataset_name = os.path.basename(xml_file).replace(".xml", "")

            for image in root.findall("image"):
                image_id = image.get("id")
                file_name = image.get("name")
                
                # Construct full image path. 
                # Note: XML 'name' attribute might be like 'NHTyp1/ATT10035.jpg'
                # We need to join this with images_dir.
                # Handle both absolute and relative paths
                if os.path.isabs(self.images_dir):
                    full_image_path = os.path.join(self.images_dir, file_name)
                else:
                    # Relative to project root
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    full_image_path = os.path.join(base_dir, self.images_dir, file_name)

                expert_score = None
                
                # Extract expert score from boxes
                # Look for a box with label="expert_score" and extract its score attribute
                for box in image.findall("box"):
                    if box.get("label") == "expert_score":
                        # Score is stored in an attribute named "score"
                        for attr in box.findall("attribute"):
                            if attr.get("name") == "score":
                                score_text = attr.text
                                if score_text:
                                    try:
                                        expert_score = int(score_text.strip())
                                    except (ValueError, AttributeError):
                                        # If conversion fails, keep as string
                                        expert_score = score_text.strip()
                                break
                        # Break out of outer loop once we found expert_score
                        if expert_score is not None:
                            break
                
                data.append({
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_path": full_image_path,
                    "expert_score": expert_score
                })

        return pd.DataFrame(data)

    @staticmethod
    def encode_image(image_path):
        """Encodes an image to base64."""
        if not os.path.exists(image_path):
            # Try fixing path if it's relative to the old 'Data' folder structure
            # e.g. if image_path is 'data/extractedimages/NHTyp1/img.jpg' but file is at
            # '/home/exouser/DSM.../Data/extractedimages/NHTyp1/img.jpg'
            # For now, we raise error or return None
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    df = loader.load_annotations()
    print(f"Loaded {len(df)} records.")
    print(df.head())

