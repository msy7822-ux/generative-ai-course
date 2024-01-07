import os
import pdf
import table
import image
import text

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credential.json"

if __name__ == "__main__":
    tables, texts = pdf.partition_pdf_by_element_type()
    tables_dict = table.summarize_tables_with_gemini(tables_list=tables)
    images_dict = image.summarize_images_with_gemini()
    text_dict = text.hypothetical_queries_with_gemini(texts_list=texts)
