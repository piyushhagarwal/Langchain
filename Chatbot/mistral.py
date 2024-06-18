import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatMistralAI(model="mistral-large-latest")

system_template = "Suppose you are a Legal Expert of India. You are asked to provide a legal opinion on the following query:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Prompt template example from string
# prompt_template = ChatPromptTemplate.from_string(
#     "Suppose you are a Legal Expert of India. You are asked to provide a legal opinion on the following query: {text}"
# )

parser = StrOutputParser()

chain = prompt_template | model | parser
text = chain.invoke({"text": "can I buy land in Kashmir?"})

print(text)