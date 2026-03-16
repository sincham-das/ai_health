import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch 
from PIL import Image


def preprocess_image(image_path):
    img=cv2.imread(image_path)
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.gaussianBlur(gray, (5,5), 0)
    
    thresh= cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    processed_path=image_path.replace("raw_images","processed")
    cv2.imwrite(processed_path, thresh)
    
    return processed_path

class TrOCREngine:
    def __init__(self):
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.processor= TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        
        self.model= VisionEncoderDecorModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(self.device)
        
    def predict(self, image):
        pixel_values=self.processor(image=image, return_tensors="pt")
        pixel_values=pixel_values.to(self.device)
        
        generate_ids=self.model.generate(pixel_values)
        text=self.processor.batch_decode(
            generate_ids, skip_special_tokens=True
        )[0]
        
        return text

def extract_text(image_path):
    processed_path=preprocesss_image(image_path)
    image=Image.open(processed_path).convert("RGB")
    
    text=engine.predict(image)
    return text.strip()

image_path="raw_images/sample.jpg"

result=extract_text(image_path)
print("OCR output:")
print(result)