from typing import Any, Dict, List

from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


def summarize_tables_with_gemini(
    tables_list: List[str],
) -> Dict[str, List[Any]]:
    table_summary_prompt_template = """
    テーブルが与えられます。
    下記に記載されている出力項目に着目して、読み取れることを出力してください。

    # 出力項目:
    - 何がまとめられているテーブルなのか
    - テーブルに記載されているキーワード
    - テーブルから読み取ることができる分析結果

    # テーブル:
    {table}
    """

    table_summary_prompt = ChatPromptTemplate.from_template(
        table_summary_prompt_template
    )

    summarize_model_name = "gemini-pro"
    summarize_model = ChatVertexAI(
        model_name=summarize_model_name,
        max_output_tokens=2048,
        temperature=0.9,
        top_p=1,
    )

    # LCELでチェーンを記述
    summarize_chain = (
        {"table": lambda x: x}
        | table_summary_prompt
        | summarize_model
        | StrOutputParser()
    )
    table_summaries = summarize_chain.batch(tables_list, config={"max_concurrency": 5})

    tables_dict = {
        "table_list": tables_list,
        "table_summaries": table_summaries,
    }

    return tables_dict
