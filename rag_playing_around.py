import os
from pathlib import Path

from llama_index.core import (StorageContext, VectorStoreIndex,load_index_from_storage)
from llama_index.readers.file import PDFReader

pdf_path = os.path.join("RAG_books", "Schiller_Mary_Stuart.pdf")

book_read = PDFReader().load_data(file=pdf_path)





