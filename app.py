from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv 


from src.helper import *
from src.prompt import system_prompt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
import os



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

embeddings = download_hugging_face_embeddings()

index_name = "medical-db"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

model_id = "google/flan-t5-base"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Create Hugging Face pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    device_map="auto"  # Uses GPU if available, else CPU
)

llm = HuggingFacePipeline(pipeline=pipe)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


@app.route("/")
def index():
    return render_template('index.html')


from flask import jsonify

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input", "")
    if not user_input:
        return jsonify({"response": "No input received."})
    app.logger.info(f"Received: {user_input}")

    result = rag_chain.invoke({"input": user_input})
    answer = result["answer"]
    app.logger.info(f"Answer: {answer}")

    # return as JSON so your JS can do response.response
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run()