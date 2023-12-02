import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from uuid import uuid4
from collections import defaultdict
from datetime import datetime

from prompts import *
import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline #BitsAndBytesConfig
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

def preprocess(dataset):
    page_content_column = "CONTEXTS"
    for split in dataset.keys():
        for contexts in dataset[split][page_content_column]:
            for sentence in contexts:
                yield Document(page_content=sentence)

def create_question2chunk(val_questions, val_contexts):
    question2chunk = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    for question, context in zip(val_questions, val_contexts):
        # split each doc into different sentences then chunk them
        context_docs = []
        for sentence in context:
            doc = Document(page_content=sentence)
            context_docs.append(doc)
        chunks = text_splitter.split_documents(context_docs)
        question2chunk[question] = chunks
    return question2chunk

def precision_at_k(r, k):
    """Score is precision @ k (This we solve for you!)
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
    Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in
    enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    #write your code here
    # print(r)
    relevant_idx = [i+1 for i in list(np.where(np.array(r)==1)[0])]
    # print(relevant_idx)
    n = sum(r)
    if n == 0: 
        return 0
    else:
        precision_k = [precision_at_k(r,pk) for pk in relevant_idx]
        avg_p = 1/n * sum(precision_k)
        return avg_p
    
def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    #write your code here
    avg_precision = [average_precision(r) for r in rs]
    n = len(rs)
    m_avg_p = 1/n * sum(avg_precision)

    return m_avg_p

# Load data
dataset = load_dataset("bigbio/pubmed_qa", cache_dir="data")
data = list(preprocess(dataset))  # 655055

print("Processing validation data")
val_dataset = dataset['validation']
val_contexts = val_dataset["CONTEXTS"][:100]
val_questions = val_dataset["QUESTION"][:100]

print("Creating dict mapping question to chunks")
question2chunk = create_question2chunk(val_questions,val_contexts)

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embedding_model = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,   
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs, 
    cache_folder="models"
)

if os.path.exists("faiss_index_pubmed"):
    print(os.path.exists("faiss_index_pubmed"))
    db = FAISS.load_local("faiss_index_pubmed", embeddings)
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    docs = text_splitter.split_documents(data)  # 676307
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index_pubmed")

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(data)
bm25_retriever.k = 10
faiss_retriever = db.as_retriever(search_kwargs={"k": 10}) 

weight_configs = [[1,0], [0.25,0.75], [0.5,0.5], [0.75,0.25], [0,1]]

now = datetime.now()
curr_time = now.strftime("%H-%M-%S")
fname = f'output_{curr_time}'

with open(fname, 'w') as f:
    for w in weight_configs:
        f.write(f'Weight config: {w}')

        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=w
        )

        all_relevance = []
        for i in tqdm(range(len(val_questions))): # for each validation query
            question = val_questions[i]
            retrieved_docs = ensemble_retriever.get_relevant_documents(question)
            retrieved_docs = retrieved_docs[:5]

            # label the retrieved list of doc as relevant or not
            bin_relevance = []
            for d in retrieved_docs:
                gt_chunks = question2chunk[question]
                if d in gt_chunks:
                    bin_relevance.append(1)
                else:
                    bin_relevance.append(0)
            all_relevance.append(bin_relevance)
            
        f.write(f'{all_relevance}')
        score = mean_average_precision(all_relevance)
        f.write(f'MAP score: {score}')