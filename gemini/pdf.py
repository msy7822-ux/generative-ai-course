import os
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

    # テーブルとテキストをリストに格納する
    tables, texts = [], []
    for element in loaded_pdf:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return tables, texts


# if __name__ == "__main__":
#     dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
#     pdf_file_path = "attention_is_all_you_need.pdf"

#     tables, texts = partition_pdf(dataset_dir=dataset_dir, pdf_file_name=pdf_file_path)
