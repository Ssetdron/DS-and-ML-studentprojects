import fitz
import os

def extract_images_from_pdf(pdf_path, output_folder="data/manual_images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"figure{page_num+1}_{img_index}.{image_ext}"
            with open(os.path.join(output_folder, image_name), "wb") as img_file:
                img_file.write(image_bytes)
    doc.close()
