import vertexai
from vertexai.preview.generative_models import GenerativeModel

PROJECT_ID = "project-id"
LOCATION = "asia-northeast1"
RESOURCE_ID = "gemini-pro"

vertexai.init(project=PROJECT_ID, location=LOCATION)


def initiate_chat_session():
    model = GenerativeModel(RESOURCE_ID)
    chat = model.start_chat()
    return chat


def multiturn_generate_content(chat, user_input):
    config = {"max_output_tokens": 2048, "temperature": 0.9, "top_p": 1}
    response = chat.send_message(user_input, generation_config=config)
    return response


chat = initiate_chat_session()
while True:
    user_input = input(">> ")
    chat_response = multiturn_generate_content(chat, user_input)
    print(chat_response)
