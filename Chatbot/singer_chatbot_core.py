# singer_chatbot_core.py

import fitz  # PyMuPDF
import numpy as np
import re
from google.genai import types

class SingerChatbot:
    def __init__(self, pdf_path, client, model="text-embedding-004"):
        self.pdf_path = pdf_path
        self.client = client
        self.model = model
        self.chunks = []
        self.embeddings = []

    # opens pdf file and extract content - first for text and second for images
    def extract_text_from_pdf(self):
        doc = fitz.open(self.pdf_path)
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        doc.close()
        return pages
    
    def extract_images_from_pdf(pdf_path, output_folder="chatbot/manual_images"):
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"figure{page_num+1}_{img_index}.{image_ext}"  # customize naming
                with open(f"{output_folder}/{image_name}", "wb") as img_file:
                    img_file.write(image_bytes)
        doc.close()

    # split text into overlapping chunks
    def chunk_text(self, pages, chunk_size=1000, overlap=200):
        words = pages.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks
    
    # send chunks to google's embedding model
    def embed_chunks(self, chunks):
        result = self.client.models.embed_content(
            model=self.model,
            contents=chunks,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings
        return [r.values for r in result]

    # include images to reference to the figures in text
    def find_figures_in_text(self, text):
        pattern = r'Fig\.?\s*\d+(\.\d+)?'
        return re.findall(pattern, text)

    # setup metod - builds the vector index by chunking and embedding pdf content
    def build_index(self):
        print("Loading and chunking PDF...")
        self.chunks = []
        self.chunk_page_map = []

        pages = self.extract_text_from_pdf() 

        for page_num, page_text in pages:
            page_chunks = self.chunk_text(page_text)
            self.chunks.extend(page_chunks)
            self.chunk_page_map.extend([page_num] * len(page_chunks))
      
        self.chunk_figures = {}
        for i, chunk in enumerate(self.chunks):
            figures = re.findall(r'(figure\s*\d+(\.\d+)?)', chunk, re.IGNORECASE)
            self.chunk_figures[i] = [f[0] for f in figures]

        print("Embedding chunks...")
        self.embeddings = self.embed_chunks(self.chunks)
        print("Embeddings generated.")

    # embeds the query
    def search(self, query, k=5):
        query_embedding = self.client.models.embed_content(
            model=self.model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings[0].values

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        scored = [(i, cosine_similarity(query_embedding, e)) for i, e in enumerate(self.embeddings)]
        top_indices = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[i], self.chunk_page_map[i]) for i, _ in top_indices]

    # the generator step - formats the prompt and returns the answer
    def ask(self, query, system_prompt, model="gemini-2.0-flash"):
        results = self.search(query)
        context = "\n".join([chunk for chunk, _ in results])
        pages = sorted(set([page for _, page in results]))
        
        user_prompt = f"The question is: {query}\n\nHere is the context:\n{context}"
        response = self.client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=[user_prompt]
        )
        return response.text, pages
