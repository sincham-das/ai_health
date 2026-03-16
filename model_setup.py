from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load processor
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Load model
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

# Optional: set padding token
model.config.pad_token_id = processor.tokenizer.pad_token_id