import fitz
import os

pdf_path = "Singer_4423_EN.pdf"
output_dir = "manual_images"

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)

# Render each page of the PDF as a high-resolution image.
# This captures all visual elements, 
# making it easier to manually identify and crop referenced figures like "Fig. 1".
for page_number in range(len(doc)):
    page = doc[page_number]
    pix = page.get_pixmap(dpi=150)  # You can increase to 200+ for better quality
    output_path = os.path.join(output_dir, f"page-{page_number + 1}.jpg")
    
    with open(output_path, "wb") as f:
        f.write(pix.tobytes("jpg"))

print(f"Rendered {len(doc)} pages to: {output_dir}")