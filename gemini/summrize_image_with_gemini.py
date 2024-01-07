import base64
import os
from typing import Any, Dict, List

from langchain.chat_models import ChatVertexAI
from langchain.schema.messages import HumanMessage


# 画像ファイルをBase64エンコードされた文字列に変換
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


# Gemini Pro Visionにて画像の説明を行い、説明結果とbase64を返却する
def summarize_images_with_gemini(image_dir: str) -> Dict[str, List[Any]]:
    image_base64_list = []
    image_summaries_list = []
    image_summary_prompt = """
    入力された画像の内容を詳細に説明してください。
    基本的には日本語で回答してほしいですが、専門用語や固有名詞を用いて説明をする際には英語のままで構いません。
    """

    for image_file_name in sorted(os.listdir(image_dir)):
        if image_file_name.endswith(".jpg"):
            image_file_path = os.path.join(image_dir, image_file_name)

            # encodeを行い、base64をリストに格納する
            image_base64 = image_to_base64(image_file_path)
            image_base64_list.append(image_base64)

            # Geminiで画像の説明を行い、結果をリストに格納する
            summarize_model_name = "gemini-pro-vision"
            summarize_model = ChatVertexAI(
                model_name=summarize_model_name,
                max_output_tokens=2048,
                temperature=0.4,
                top_p=1,
                top_k=32,
            )

            text_message = {"type": "text", "text": image_summary_prompt}
            image_message = {
                "type": "image_url",
                "image_url": {"url": image_file_path},
            }
            response = summarize_model(
                [HumanMessage(content=[text_message, image_message])]
            )
            image_summaries_list.append(response.content)
            images_dict = {
                "image_list": image_base64_list,
                "image_summaries": image_summaries_list,
            }

    return images_dict


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

    images_dict = summarize_images_with_gemini(
        image_dir=os.path.join(dataset_dir, "images")
    )
