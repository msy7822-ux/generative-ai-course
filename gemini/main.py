import os
import pdf
import table
import image
import text
import vector_db
import rag_functions

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# Chainを作成、実行する
def multimodal_rag(retriever: MultiVectorRetriever, question: str) -> str:
    chain = (
        {
            "context": retriever | RunnableLambda(rag_functions.split_data_type),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(rag_functions.generate_prompt)
        | RunnableLambda(rag_functions.model_selection)
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer


GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credential.json"

if __name__ == "__main__":
    tables, texts = pdf.partition_pdf_by_element_type()
    tables_dict = table.summarize_tables_with_gemini(tables_list=tables)
    images_dict = image.summarize_images_with_gemini()
    text_dict = text.hypothetical_queries_with_gemini(texts_list=texts)

    multivector_retriever = vector_db.import_data_to_vector_store(
        texts_dict=text_dict, tables_dict=tables_dict, images_dict=images_dict
    )

    # question_1 = "Attentionの論文の著者は誰ですか？また、所属はどこですか？"
    question_1 = "請求先の企業はどこですか？"
    answer_1 = multimodal_rag(multivector_retriever, question_1)
    print(answer_1)

    # question_2 = "Transformerのアーキテクチャとはどのようなものですか？"
    question_2 = "適格請求書発行事業者の登録番号を教えてください。"
    answer_2 = multimodal_rag(multivector_retriever, question_2)
    print(answer_2)
