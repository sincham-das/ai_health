import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class PrescriptionDataset(Dataset):

    def __init__(self, csv_file, image_folder, processor):

        # Load CSV file
        self.data = pd.read_csv(csv_file)
        print(self.data.columns)

        # Image folder path
        self.image_folder = image_folder

        # TrOCR processor
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # Build image path
        image_path = os.path.join(self.image_folder, row["filename"])

        # Ground truth text
        text = row["label"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Convert image → pixel values
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze()

        # Convert text → token ids
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).input_ids.squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }