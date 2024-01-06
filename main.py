import os
from dotenv import load_dotenv
import openai
import utils

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Fileのアップロード
file = openai.files.create(
    file=open("./attention_is_all_you_need.pdf", "rb"),
    purpose="assistants",
)

assistant = openai.beta.assistants.create(
    name="Prompt Engineer Bot",
    description="Transformerモデルについて詳しいアシスタントです。",
    model="gpt-4-1106-preview",
    instructions="あなたは、Transformerモデルについて詳しいアシスタントです。Attentionに関するPDFの教材を参考にして、回答してください。",
    tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
    file_ids=[file.id],
)

thread = openai.beta.threads.create()
thread_id = thread.id

message = openai.beta.threads.messages.create(
    thread_id=thread_id,
    role="user",
    content="Transformerモデルの基本的な構造を全て日本語で教えてください。",
)

run = openai.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant.id,
)

utils.wait_on_run(run, thread)
messages = openai.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)

for message in messages.data:
    print(message.content[0].text.value)
