import tiktoken
import openai
from rich import print

from xiaogpt.bot.base_bot import BaseBot


class AzureOpenAIBot(BaseBot):
    max_response_tokens = 250
    token_limit = 4000
    system_msg = {"role": "system", "content": "你是一个语音助手，你的名字叫小爱同学，你的回答需要方便被朗读。"}

    def __init__(self, openai_key, engine, api_base=None, proxy=None):
        self.history = []
        self.history.append(self.system_msg)
        self.engine = engine
        openai.api_key = openai_key
        if api_base:
            openai.api_base = api_base
        if proxy:
            openai.proxy = proxy
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"

    def num_tokens_from_messages(self, messages):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    async def ask(self, query, **options):
        self.history.append({"role": "user", "content": f"{query}"})
        history_tokens = self.num_tokens_from_messages(self.history)

        while history_tokens > self.token_limit:
            del self.history[1]
            history_tokens = self.num_tokens_from_messages(self.history)

        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.history,
            temperature=0.5,
            max_tokens=self.max_response_tokens,
            stream=False,
        )
        message = response["choices"][0]["message"]["content"]
        print(message + "\n")
        self.history.append({"role": "assistant", "content": message})
        return message

    async def ask_stream(self, query, **options):
        self.history.append({"role": "user", "content": f"{query}"})
        history_tokens = self.num_tokens_from_messages(self.history)

        while history_tokens > self.token_limit:
            del self.history[1]
            history_tokens = self.num_tokens_from_messages(self.history)

        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.history,
            temperature=0.5,
            max_tokens=self.max_response_tokens,
            stream=True,
        )
        message = ""
        for event in response:
            chunk_message = event["choices"][0]["delta"]
            if "content" not in chunk_message:
                continue
            message += chunk_message["content"]
            yield chunk_message["content"]
        self.history.append({"role": "assistant", "content": message})
        print(message + "\n")
