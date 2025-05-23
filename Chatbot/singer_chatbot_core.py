# singer_chatbot_core.py

import fitz  # PyMuPDF
import numpy as np
from google.genai import types

class SingerChatbot:
    def __init__(self, pdf_path, client, model="text-embedding-004"):
        self.pdf_path = pdf_path
        self.client = client
        self.model = model
        self.chunks = []
        self.embeddings = []

    # opens pdf file and extract content
    def extract_text_from_pdf(self):
        doc = fitz.open(self.pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return pages

    # split text into overlapping chunks
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        words = text.split()
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

    # setup metod - builds the vector index by chunking and embedding pdf content
    def build_index(self):
        print("Loading and chunking PDF...")
        pages = self.extract_text_from_pdf()
        full_text = "\n".join(pages)
        self.chunks = self.chunk_text(full_text)
        print(f"Created {len(self.chunks)} chunks.")
        
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
        return [self.chunks[i] for i, _ in top_indices]

    # the generator step - formats the prompt and returns the answer
    def ask(self, query, system_prompt, model="gemini-2.0-flash"):
        context = "\n".join(self.search(query))
        user_prompt = f"The question is: {query}\n\nHere is the context:\n{context}"
        response = self.client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=[user_prompt]
        )
        return response.text
