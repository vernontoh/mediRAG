import os
import gc
import torch
import numpy as np
from prompts import *
from utils import *
from config import HF_TOKEN
import evaluate
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain import hub
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFaceHub
from FlagEmbedding import FlagReranker
from langchain.retrievers import EnsembleRetriever


model_name= 'mistralai/Mistral-7B-Instruct-v0.1'

# load tokeniser
tokenizer = AutoTokenizer.from_pretrained(model_name,   cache_dir="models")

# Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
    
# Loading Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    cache_dir="models",
    # device_map = "auto"
)

#load_data
dataset = load_dataset("bigbio/pubmed_qa", cache_dir="data")
data = list(preprocess(dataset))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# loading best Retriever
repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 64}
)

embedding_model1 = "pritamdeka/S-PubMedBert-MS-MARCO"
embedding_model2 = "BAAI/bge-large-en-v1.5"

model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings1 = HuggingFaceEmbeddings(
    model_name=embedding_model1,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="models"
)

if os.path.exists(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{1024}_co{128}"):
    print("Loading existing FAISS index...")
    db_pubmed = FAISS.load_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{1024}_co{128}", embeddings1)
else:
    print("Creating FAISS index...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    docs = text_splitter.split_documents(data) 
    db_pubmed = FAISS.from_documents(docs, embeddings1)
    db_pubmed.save_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{1024}_co{128}")


embeddings2 = HuggingFaceEmbeddings(
    model_name=embedding_model2,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="models"
)

if os.path.exists(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{1024}_co{128}"):
    print("Loading existing FAISS index...")
    db_bge = FAISS.load_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{1024}_co{128}", embeddings2)
else:
    print("Creating FAISS index...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    docs = text_splitter.split_documents(data) 
    db_bge = FAISS.from_documents(docs, embeddings2)
    db_bge.save_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{1024}_co{128}")

pubmed_retriever = db_pubmed.as_retriever(search_kwargs={"k": 10}) 
bge_retriever = db_bge.as_retriever(search_kwargs={"k": 10})

# Initialising Retriever
ensemble_retriever = EnsembleRetriever(
                retrievers=[pubmed_retriever, bge_retriever], weights=[0.75,0.25]
            )
reranker = FlagReranker('BAAI/bge-reranker-large')

def retriever(question):
    expanded_question = query_expansion(llm, question, type="prompt")
    retrieved_docs = ensemble_retriever.get_relevant_documents(expanded_question)
    retrieved_docs = retrieved_docs[:15]
    retrieved_docs = rerank_topk(reranker, question, retrieved_docs)
    retrieved_docs = retrieved_docs[:5]
    return retrieved_docs

# Building Pipeline
text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            max_new_tokens=300,
            do_sample=False,
        )

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template= prompt_templates['retrieval']['cot'],
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | mistral_llm
    | StrOutputParser()
)

def QA(question):
    result = chain.invoke(question)
    result = find_citations([result], db_bge)[0]
    return result

if __name__ == "__main__":
    print("#"*20)
    while True:
        question = input("Enter a Biomedical Question:")
        if question == 'x':
            break
        try:
            Answer = QA(question)
            print(f"Answer: {Answer}")
        except Exception as error:
            print(error)