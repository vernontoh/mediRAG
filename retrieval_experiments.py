import os
import time
import torch
import argparse
from utils import *
from tqdm import tqdm as tqdm

from datasets import load_dataset
from FlagEmbedding import FlagReranker
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--task', type=str, required=True, 
        help="Evaluation Task. chunk, single, multi_dense, hybrid, rerank, query_expand"
    )
    parser.add_argument(
        '--chunk_size', type=int, default=1024, 
        help="Document chunk size"
    )
    parser.add_argument(
        '--chunk_overlap', type=int, default=128,
        help="Document chunk overlap"
    )
    parser.add_argument(
        '--hf_token', type=str, default="",
        help="Huggingface API token"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print("Loading PubMedQA...")
    dataset = load_dataset("bigbio/pubmed_qa", cache_dir="data")
    data = list(preprocess(dataset))  # 655055

    print("Processing validation data...")
    val_dataset = dataset['validation']
    val_contexts = val_dataset["CONTEXTS"][:100]
    val_questions = val_dataset["QUESTION"][:100]

    out_path = f'evaluation_outputs/{time.strftime("%Y%m%d-%H%M%S")}_{args.task}'

    if args.task == "chunk":
        print("#" * 80)
        print("Running Experiment: chunk")
        print("#" * 80)

        embedding_model = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,   
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs, 
            cache_folder="models"
        )

        for chunk_size_overlap in [(512, 64), (1024, 128)]:
            chunk_size, chunk_overlap = chunk_size_overlap

            print("Creating dict mapping question to chunks...")
            question2chunk = create_question2chunk(
                val_questions, 
                val_contexts, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
                )

            if os.path.exists(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{chunk_size}_co{chunk_overlap}"):
                print("Loading existing FAISS index...")
                db = FAISS.load_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{chunk_size}_co{chunk_overlap}", embeddings)
            else:
                print("Creating FAISS index...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs = text_splitter.split_documents(data) 
                db = FAISS.from_documents(docs, embeddings)
                db.save_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{chunk_size}_co{chunk_overlap}")

            retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

            print("Starting Experiment...")
            with open(out_path, 'a+') as f:
                f.write(f'{embedding_model} chunk size {chunk_size} chunk overlap {chunk_overlap} | ')
                
                all_relevance = []
                for i in tqdm(range(len(val_questions))): # for each validation query
                    question = val_questions[i]
                    retrieved_docs = retriever.get_relevant_documents(question)
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
                    
                # f.write(f'{all_relevance}')
                score = mean_average_precision(all_relevance)
                f.write(f'MAP score: {score}\n')
                f.flush()
        

    elif args.task == "single":
        print("#" * 80)
        print("Running Experiment: single")
        print(f"Chunk size = {args.chunk_size}")
        print(f"Chunk overlap = {args.chunk_overlap}")
        print("#" * 80)

        print("Creating dict mapping question to chunks...")
        question2chunk = create_question2chunk(
            val_questions, 
            val_contexts, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
            )

        embedding_models = [
            "BAAI/bge-large-en-v1.5", 
            "pritamdeka/S-PubMedBert-MS-MARCO"
            ]
        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        for embedding_model in embedding_models:
            print(f"Evaluating using {embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,   
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs, 
                cache_folder="models"
            )

            if os.path.exists(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
                print("Loading existing FAISS index...")
                db = FAISS.load_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings)
            else:
                print("Creating FAISS index...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                docs = text_splitter.split_documents(data) 
                db = FAISS.from_documents(docs, embeddings)
                db.save_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")

            retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

            print("Starting Experiment...")
            with open(out_path, 'a+') as f:
                f.write(f'{embedding_model} | ')
                
                all_relevance = []
                for i in tqdm(range(len(val_questions))): # for each validation query
                    question = val_questions[i]
                    retrieved_docs = retriever.get_relevant_documents(question)
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
                    
                # f.write(f'{all_relevance}')
                score = mean_average_precision(all_relevance)
                f.write(f'MAP score: {score}\n')
                f.flush()


    elif args.task == "multi_dense":
        print("#" * 80)
        print("Running Experiment: multi_dense")
        print(f"Chunk size = {args.chunk_size}")
        print(f"Chunk overlap = {args.chunk_overlap}")
        print("#" * 80)

        print("Creating dict mapping question to chunks...")
        question2chunk = create_question2chunk(
            val_questions, 
            val_contexts, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
            )
        
        embedding_model1 = "pritamdeka/S-PubMedBert-MS-MARCO"
        embedding_model2 = "BAAI/bge-large-en-v1.5"

        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings1 = HuggingFaceEmbeddings(
            model_name=embedding_model1,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_pubmed = FAISS.load_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings1)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_pubmed = FAISS.from_documents(docs, embeddings1)
            db_pubmed.save_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")


        embeddings2 = HuggingFaceEmbeddings(
            model_name=embedding_model2,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_bge = FAISS.load_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings2)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_bge = FAISS.from_documents(docs, embeddings2)
            db_bge.save_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")

        pubmed_retriever = db_pubmed.as_retriever(search_kwargs={"k": 10}) 
        bge_retriever = db_bge.as_retriever(search_kwargs={"k": 10}) 

        weight_configs = [[0.25,0.75], [0.5,0.5], [0.75,0.25]]

        print("Starting Experiment...")
        with open(out_path, 'a+') as f:
            for w in weight_configs:
                print(f"Evaluating with S-PubMedBert-MS-MARCO weight = {w[0]} and bge-large-en-v1.5 weight = {w[-1]}")
                f.write(f'Evaluating with S-PubMedBert-MS-MARCO weight = {w[0]} and bge-large-en-v1.5 weight = {w[-1]} | ')

                # initialize the ensemble retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[pubmed_retriever, bge_retriever], weights=w
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
                    
                # f.write(f'{all_relevance}')
                score = mean_average_precision(all_relevance)
                f.write(f'MAP score: {score}\n')
                f.flush()

        

    elif args.task == "hybrid":
        print("#" * 80)
        print("Running Experiment: hybrid")
        print(f"Chunk size = {args.chunk_size}")
        print(f"Chunk overlap = {args.chunk_overlap}")
        print("#" * 80)

        print("Creating dict mapping question to chunks...")
        question2chunk = create_question2chunk(
            val_questions, 
            val_contexts, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
            )

        embedding_models = [
            "BAAI/bge-large-en-v1.5", 
            "pritamdeka/S-PubMedBert-MS-MARCO"
            ]
        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        for embedding_model in embedding_models:
            print(f"Evaluating using {embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,   
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs, 
                cache_folder="models"
            )

            if os.path.exists(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
                print("Loading existing FAISS index...")
                db = FAISS.load_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings)
            else:
                print("Creating FAISS index...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                docs = text_splitter.split_documents(data) 
                db = FAISS.from_documents(docs, embeddings)
                db.save_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")

            print("Initialize the bm25 retriever and faiss retriever...")
            bm25_retriever = BM25Retriever.from_documents(data)
            bm25_retriever.k = 10
            faiss_retriever = db.as_retriever(search_kwargs={"k": 10}) 

            weight_configs = [[1,0], [0.25,0.75], [0.5,0.5], [0.75,0.25], [0,1]]

            print("Starting Experiment...")
            with open(out_path, 'a+') as f:
                for w in weight_configs:
                    print(f"Evaluating with BM25 weight = {w[0]} and {embedding_model} weight = {w[-1]}")
                    f.write(f"Evaluating with BM25 weight = {w[0]} and {embedding_model} weight = {w[-1]} | ")

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
                        
                    # f.write(f'{all_relevance}')
                    score = mean_average_precision(all_relevance)
                    f.write(f'MAP score: {score}\n')
                    f.flush()

    elif args.task == "rerank":
        print("#" * 80)
        print("Running Experiment: rerank")
        print(f"Chunk size = {args.chunk_size}")
        print(f"Chunk overlap = {args.chunk_overlap}")
        print("#" * 80)

        print("Creating dict mapping question to chunks...")
        question2chunk = create_question2chunk(
            val_questions, 
            val_contexts, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
            )
        
        embedding_model1 = "pritamdeka/S-PubMedBert-MS-MARCO"
        embedding_model2 = "BAAI/bge-large-en-v1.5"

        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings1 = HuggingFaceEmbeddings(
            model_name=embedding_model1,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_pubmed = FAISS.load_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings1)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_pubmed = FAISS.from_documents(docs, embeddings1)
            db_pubmed.save_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")


        embeddings2 = HuggingFaceEmbeddings(
            model_name=embedding_model2,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_bge = FAISS.load_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings2)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_bge = FAISS.from_documents(docs, embeddings2)
            db_bge.save_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")

        pubmed_retriever = db_pubmed.as_retriever(search_kwargs={"k": 10}) 
        bge_retriever = db_bge.as_retriever(search_kwargs={"k": 10}) 

        print("Starting Experiment...")
        with open(out_path, 'a+') as f:

            # initialize the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[pubmed_retriever, bge_retriever], weights=[0.75,0.25]
            )

            reranker = FlagReranker('BAAI/bge-reranker-large')

            for topk in [5, 10, 15]:
                f.write(f"Reranking {topk} documents using bge-reranker-large | ")
                all_relevance = []
                for i in tqdm(range(len(val_questions))): # for each validation query
                    question = val_questions[i]
                    retrieved_docs = ensemble_retriever.get_relevant_documents(question)
                    retrieved_docs = retrieved_docs[:topk]
                    retrieved_docs = rerank_topk(reranker, question, retrieved_docs)
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

                # f.write(f'{all_relevance}')
                score = mean_average_precision(all_relevance)
                f.write(f'MAP score: {score}\n')
                f.flush()

    elif args.task == "query_expand":
        print("#" * 80)
        print("Running Experiment: query_expand")
        print(f"Chunk size = {args.chunk_size}")
        print(f"Chunk overlap = {args.chunk_overlap}")
        print("#" * 80)

        print("Creating dict mapping question to chunks...")
        question2chunk = create_question2chunk(
            val_questions, 
            val_contexts, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
            )
        
        embedding_model1 = "pritamdeka/S-PubMedBert-MS-MARCO"
        embedding_model2 = "BAAI/bge-large-en-v1.5"

        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings1 = HuggingFaceEmbeddings(
            model_name=embedding_model1,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_pubmed = FAISS.load_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings1)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_pubmed = FAISS.from_documents(docs, embeddings1)
            db_pubmed.save_local(f"index/pubmedqa_{embedding_model1.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")


        embeddings2 = HuggingFaceEmbeddings(
            model_name=embedding_model2,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}"):
            print("Loading existing FAISS index...")
            db_bge = FAISS.load_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}", embeddings2)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            docs = text_splitter.split_documents(data) 
            db_bge = FAISS.from_documents(docs, embeddings2)
            db_bge.save_local(f"index/pubmedqa_{embedding_model2.split('/')[-1]}_cs{args.chunk_size}_co{args.chunk_overlap}")

        pubmed_retriever = db_pubmed.as_retriever(search_kwargs={"k": 10}) 
        bge_retriever = db_bge.as_retriever(search_kwargs={"k": 10}) 

        print("Starting Experiment...")
        with open(out_path, 'a+') as f:

            # initialize the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[pubmed_retriever, bge_retriever], weights=[0.75,0.25]
            )

            os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_token
            repo_id = "google/flan-t5-xxl"
            llm = HuggingFaceHub(
                repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 256}
            )

            reranker = FlagReranker('BAAI/bge-reranker-large')

            for type_prompt in [1, 2]:
                for wiki in [True, False]:
                    f.write(f'Query Expansion with flan-t5-xxl with type prompt {type_prompt} and wiki {wiki} | ')
                    all_relevance = []
                    for i in tqdm(range(len(val_questions))): # for each validation query
                        time.sleep(5)
                        question = val_questions[i]
                        expanded_question = query_expansion(llm, question, type_prompt=type_prompt, wiki=wiki)
                        retrieved_docs = ensemble_retriever.get_relevant_documents(expanded_question)
                        retrieved_docs = retrieved_docs[:15]
                        retrieved_docs = rerank_topk(reranker, question, retrieved_docs)
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

                    # f.write(f'{all_relevance}')
                    score = mean_average_precision(all_relevance)
                    f.write(f'MAP score: {score}\n')
                    f.flush()


if __name__ == '__main__':
    torch.cuda.set_device(0)

    main()