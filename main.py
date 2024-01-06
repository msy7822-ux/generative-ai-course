import os
from dotenv import load_dotenv
import openai

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Fileのアップロード
uploaded_file = openai.files.create(
    file=open("attention_is_all_you_need.pdf", "rb"),
    purpose="assistants",
)

import time


# run, threadが非同期関数？
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


# assistant = openai.beta.assistants.create(
#     name="Prompt Engineer Bot",
#     description="Transformerモデルについて詳しいアシスタントです。",
#     model="gpt-4-1106-preview",
#     instructions="あなたは、Transformerモデルについて詳しいアシスタントです。Attentionに関するPDFの教材を参考にして、回答してください。",
#     tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
#     file_ids=[file.id],
# )

assistant = openai.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="gpt-4-1106-preview",
)

import json


def show_json(obj):
    print(json.loads(obj.model_dump_json()))


thread = openai.beta.threads.create()
thread_id = thread.id

# openai.beta.threads.messages.create(
#     thread_id=thread_id,
#     role="user",
#     content="PDFのタイトルを教えてください。",
# )

message = openai.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)
# show_json(message)

run = openai.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant.id,
)

wait_on_run(run, thread)
messages = openai.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)

for message in messages.data:
    print(message.content[0].text.value)

# run_retrieve = openai.beta.threads.runs.retrieve(
#     thread_id=thread_id,
#     run_id=run.id,
# )


# messages = openai.beta.threads.messages.list(thread_id=thread_id)
