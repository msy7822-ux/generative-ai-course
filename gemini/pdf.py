from typing import List, Tuple
from unstructured.partition.pdf import partition_pdf

path = "/Users/msy/ai/generative-ai-course/"


def partition_pdf_by_element_type() -> Tuple[List[str], List[str]]:
    pdf_file_name = "datasets/attention_is_all_you_need.pdf"

    loaded_pdf = partition_pdf(
        filename=path + pdf_file_name,
        languages=["jpn", "eng"],
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path + "gemini/public/output/",
    )

    tables, texts = [], []
    for element in loaded_pdf:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return tables, texts
