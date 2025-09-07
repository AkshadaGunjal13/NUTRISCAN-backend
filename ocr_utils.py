import pytesseract
from PIL import Image
import re



# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # OCR extraction
    text = pytesseract.image_to_string(img)

    # Clean text
    text = text.lower()
    img = img.convert("RGB")
    text = re.sub(r'[^a-z, ]', '', text)
    return text



