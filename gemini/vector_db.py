import uuid
from typing import Any, Dict, List

from langchain.embeddings import VertexAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_core.documents import Document


def import_data_to_vector_store(
    texts_dict: Dict[str, List[Any]],
    tables_dict: Dict[str, List[Any]],
    images_dict: Dict[str, List[Any]],
) -> MultiVectorRetriever:
    embedding_model_name = "textembedding-gecko-multilingual@001"
    # embedding_model_name = "textembedding-gecko@003"
    embedding_function = VertexAIEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma(
        collection_name="gemini-pro-multi-rag",
        embedding_function=embedding_function,
    )

    # 元の文章を保存するためのストレージ
    store = InMemoryStore()
    id_key = "doc_id"

    # Retrieverの作成
    multivector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 6},
    )

    # テキストデータをembedding、vectorstoreに格納する
    doc_ids = [str(uuid.uuid4()) for _ in texts_dict["texts_list"]]
    # チャンクを保存する
    for i, s in enumerate(texts_dict["texts_list"]):
        if s != "":
            multivector_retriever.vectorstore.add_documents(
                [Document(page_content=s, metadata={id_key: doc_ids[i]})]
            )
    # テキストチャンクとidを紐づける
    multivector_retriever.docstore.mset(list(zip(doc_ids, texts_dict["texts_list"])))
    print("### Text Data Stored! ###")

    # 想定質問を保存する
    doc_summary_ids = [str(uuid.uuid4()) for _ in texts_dict["texts_list"]]
    for i, s in enumerate(texts_dict["text_summaries"]):
        if s != "":
            multivector_retriever.vectorstore.add_documents(
                [Document(page_content=s, metadata={id_key: doc_summary_ids[i]})]
            )
    # テキストチャンクとidを紐づける
    multivector_retriever.docstore.mset(
        list(zip(doc_summary_ids, texts_dict["texts_list"]))
    )
    print("### Hypothetical Queries Data Stored! ###")

    # テーブルデータの説明をembedding、vectorstoreに格納する
    table_ids = [str(uuid.uuid4()) for _ in tables_dict["table_list"]]
    # テーブルの説明を保存する
    for i, s in enumerate(tables_dict["table_summaries"]):
        multivector_retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: table_ids[i]})]
        )
    # tablesを保存、とidを紐づける
    multivector_retriever.docstore.mset(list(zip(table_ids, tables_dict["table_list"])))
    print("### Table Data Stored! ###")

    # 画像データの説明をembedding、vectorstoreに格納する
    img_ids = (
        [str(uuid.uuid4()) for _ in images_dict["image_list"]]
        if "image_list" in images_dict.keys()
        else []
    )
    # 画像の説明を保存する
    for i, s in enumerate(
        images_dict["image_summaries"]
        if "image_summaries" in images_dict.keys()
        else []
    ):
        multivector_retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: img_ids[i]})]
        )

    # imagesを保存、とidを紐づける
    multivector_retriever.docstore.mset(
        list(
            zip(
                img_ids,
                images_dict["image_list"] if "image_list" in images_dict.keys() else [],
            )
        )
    )
    print("### Image Data Stored! ###")

    return multivector_retriever
