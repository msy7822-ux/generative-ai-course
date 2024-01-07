from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import Any, Dict, List


def hypothetical_queries_with_gemini(
    texts_list: List[str],
) -> Dict[str, List[Any]]:
    text_summary_prompt_template = """
    テキストチャンクが与えられます。
    そのチャンクに対して、想定される質問を1つ考えてください。
    下記の制約条件を厳格に守ってください。

    # 制約条件:
    - あなたが考えた質問のみを出力してください
    - 質問を考える際には、質問例を参考にしてください
    - 1つのテキストチャンクに対して、1つの回答を出力してください

    # 質問例:
    - MultiVectorRetrieverとはどのようなものですか？
    - 「Vertex AI(Gemini API)でGemini Proを試す」というブログの著者は誰ですか？
    - Vertex AIのGemini APIではどのようなことができますか？

    # テキストチャンク:
    {text}
    """

    text_summary_prompt = ChatPromptTemplate.from_template(text_summary_prompt_template)

    summarize_model_name = "gemini-pro"
    summarize_model = ChatVertexAI(
        model_name=summarize_model_name,
        max_output_tokens=2048,
        temperature=0.9,
        top_p=1,
    )

    # LCELでチェーンを記述
    summarize_chain = (
        {"text": lambda x: x}
        | text_summary_prompt
        | summarize_model
        | StrOutputParser()
    )
    text_summaries = summarize_chain.batch(texts_list, config={"max_concurrency": 5})
    texts_dict = {
        "texts_list": texts_list,
        "text_summaries": text_summaries,
    }

    return texts_dict
