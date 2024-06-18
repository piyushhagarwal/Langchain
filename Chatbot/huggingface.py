import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

import huggingface_hub
huggingface_hub.login(token=os.getenv("HUGGINGFACE_API_KEY"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

model_id = "google/gemma-2b"

# Load the base language model directly, not through LangChain
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create a pipeline using the base model
pipe = pipeline("text-generation", model=model,tokenizer=tokenizer,  max_new_tokens=10)

# Wrap the pipeline in a LangChain HuggingFacePipeline
hf_pipeline = HuggingFacePipeline(pipeline=pipe)

system_template = "You are a chat bot"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

chain = prompt_template | hf_pipeline | parser
text = chain.invoke({"text": "Hello"})

print(text)