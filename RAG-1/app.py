import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set API keys for LangChain and Mistral from environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# Import necessary classes for the model and output parsing
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

# Initialize the ChatMistralAI model with the specified version
model = ChatMistralAI(model="mistral-large-latest")

# Use BeautifulSoup (bs4) to filter out only the post title, headers, and content
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Define a SoupStrainer to filter specific parts of the HTML
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

# Load the post content from the URL using the specified filter
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# Uncomment the following line to print the post content
# print(docs[0].page_content)

# Split the document into chunks of 1000 characters with 200 characters overlap
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Uncomment the following lines to print the split content
# for split in all_splits:
#     print(split.page_content)
#     print("--------------------------------------------------")

# Generate embeddings for each split using HuggingFace's sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings()

# Store these embeddings in a FAISS vector database
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings_model)

# Retrieve the most similar documents to a query
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Retrieve documents related to the query
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# Uncomment the following line to print the retrieved documents
# print(retrieved_docs)

# Import necessary classes for creating a prompt template and chains
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Define a custom prompt template for the RAG (Retrieve and Generate) chain
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {input}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Create a chain that combines retrieved documents and generates an answer
question_answer_chain = create_stuff_documents_chain(model, custom_rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke the chain with a query and get the response
response = rag_chain.invoke({"input": "What is Task Decomposition?"})

# Print the answer from the response
print(response["answer"])
