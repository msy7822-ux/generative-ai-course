import os
import pdf

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if __name__ == "__main__":
    tables, texts = pdf.partition_pdf_by_element_type()
    print(tables, texts)
