import argparse
import os
import gc
import torch
import numpy as np
from prompts import *
from utils import *
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='A script to evaluate the performance of different LLM prompts and retrieval query approaches')

    parser.add_argument('--retrieval_type', choices=["base", "best"], default="base")

    parser.add_argument('--prompt_type', choices=["cot" , "base" , "one-shot"], default="base")

    parser.add_argument('--approach', choices=["vanilla" , "citations" , "retrieval", "stepback"], default="vanilla")
    
    parser.add_argument('--n_samples' , type=int , default=100)

    parser.add_argument('--save_dir', type=str, default="evaluation_outputs")

    parser.add_argument('--hf_token', type=str, default="", help="Huggingface API token")

    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose mode')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print(args)

    # Load Model
    if args.verbose:
        print("Instantiating Model and Tokeniser")
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

    # Load data
    if args.verbose:
        print("Loading Dataset")
    dataset = load_dataset("bigbio/pubmed_qa", cache_dir="data")
    data = list(preprocess(dataset))

    if args.retrieval_type == "base":
        # Setting up Faiss Vectorstore
        if args.verbose:
            print("Instantiating Vectorstore")
        embedding_model = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device':'cuda'}
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,   
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs, 
            cache_folder="models"
        )

        if os.path.exists(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{1024}_co{128}"):
            print("Loading existing FAISS index...")
            db = FAISS.load_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{1024}_co{128}", embeddings)
        else:
            print("Creating FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
            docs = text_splitter.split_documents(data) 
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(f"index/pubmedqa_{embedding_model.split('/')[-1]}_cs{1024}_co{128}")


        # Initialising Retriever
        if args.verbose:
            print("Instantiating Retriever")
            
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3}
        )

        # Loading Questions and Ground truth Answers
        questions = dataset['train'][2:2+args.n_samples]["QUESTION"]
        contexts = dataset['train'][2:2+args.n_samples]["CONTEXTS"]
        long_answers = dataset['train'][2:2+args.n_samples]["LONG_ANSWER"]
        final_decisions = dataset['train'][2:2+args.n_samples]["final_decision"]

        # Initialising Pipeline
        if args.verbose:
            print("Instantiating Text Generation Pipeline and LLM Chain")

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            max_new_tokens=300,
            do_sample=False,
        )

        mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        if args.approach in ["vanilla", "citations"]:

            prompt = PromptTemplate(
                input_variables=["question"],
                template= prompt_templates['vanilla'][args.prompt_type],
            )

            chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | mistral_llm
                | StrOutputParser()
            )
            
        elif args.approach == 'retrieval':
            # Create prompt from prompt template 
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template= prompt_templates['retrieval'][args.prompt_type],
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | mistral_llm
                | StrOutputParser()
            )
            
        else:
            stepback_prompt = PromptTemplate(
                input_variables=["question"],
                template=STEPBACK_PROMPT,
            )

            response_prompt = PromptTemplate(
                input_variables=["context","step_back_context","question"],
                template=prompt_templates[args.approach][args.prompt_type],
            )

            chain = (
                {
                    # Retrieve context using the normal question
                    "context": RunnablePassthrough() | retriever,
                    # Retrieve context using the step-back question
                    "step_back_context": {"question": RunnablePassthrough()} |stepback_prompt | mistral_llm | StrOutputParser() | retriever,
                    # Pass on the question
                    "question": RunnablePassthrough(),
                }
                | response_prompt
                | mistral_llm
                | StrOutputParser()
            )


    elif args.retrieval_type == "best":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_token
        repo_id = "google/flan-t5-xxl"
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 64}
        )
        if args.verbose:
            print("Instantiating Vectorstore")
        embedding_model1 = "pritamdeka/S-PubMedBert-MS-MARCO"
        embedding_model2 = "BAAI/bge-large-en-v1.5"

        model_kwargs = {'device':'cpu'}  # cuda stepback OOM
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
        if args.verbose:
            print("Instantiating Retriever")
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

        # Loading Questions and Ground truth Answers
        questions = dataset['train'][2:2+args.n_samples]["QUESTION"]
        contexts = dataset['train'][2:2+args.n_samples]["CONTEXTS"]
        long_answers = dataset['train'][2:2+args.n_samples]["LONG_ANSWER"]
        final_decisions = dataset['train'][2:2+args.n_samples]["final_decision"]

        # Initialising Pipeline
        if args.verbose:
            print("Instantiating Text Generation Pipeline and LLM Chain")

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            max_new_tokens=300,
            do_sample=False,
        )

        mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        if args.approach in ["vanilla", "citations"]:

            prompt = PromptTemplate(
                input_variables=["question"],
                template= prompt_templates['vanilla'][args.prompt_type],
            )

            chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | mistral_llm
                | StrOutputParser()
            )
            
        elif args.approach == 'retrieval':
            # Create prompt from prompt template 
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template= prompt_templates['retrieval'][args.prompt_type],
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | mistral_llm
                | StrOutputParser()
            )

        else:
            stepback_prompt = PromptTemplate(
                input_variables=["question"],
                template=STEPBACK_PROMPT,
            )

            response_prompt = PromptTemplate(
                input_variables=["context","step_back_context","question"],
                template=prompt_templates[args.approach][args.prompt_type],
            )

            chain = (
                {
                    # Retrieve context using the normal question
                    "context": RunnablePassthrough() | retriever,
                    # Retrieve context using the step-back question
                    "step_back_context": {"question": RunnablePassthrough()} |stepback_prompt | mistral_llm | StrOutputParser() | retriever,
                    # Pass on the question
                    "question": RunnablePassthrough(),
                }
                | response_prompt
                | mistral_llm
                | StrOutputParser()
            )

    # Getting Outputs
    if args.verbose:
        print("Generating Outputs")
    results = []
    for query in tqdm(questions):    
        if args.verbose:
            print(f"Question: {query}")
        while True:
            try:
                result = chain.invoke(query)
                break
            except Exception as error:
                print(error)
                time.sleep(30)

        if args.verbose:
            print(f"Generated Output: {result}")
        results.append(result)
    if args.approach == 'citations':
        if args.verbose:
            print(f"Generating Citations")
        results = find_citations(results, db)
    predictions = results
    if args.approach == 'citations':
        predictions = clean_cit(predictions)

    # Evaluation
    del model, text_generation_pipeline, mistral_llm, chain, retriever
    gc.collect()
    torch.cuda.empty_cache()
    # BLEU
    bleu = evaluate.load("bleu", cache_dir="evaluation_metrics")
    bleu_score = bleu.compute(predictions=predictions, references=long_answers)
    print(f"BLEU Score: {bleu_score}")

    # BERT Score
    bertscore = evaluate.load("bertscore", cache_dir="evaluation_metrics")

    bert_score = bertscore.compute(predictions=predictions, references=long_answers , lang="en", batch_size =1)
    bert_score = {key: np.mean(value) if key!= "hashcode" else value for key, value in bert_score.items()}
    print(f"BERTScore: {bert_score}")

    del bertscore
    torch.cuda.empty_cache()

    # Perplexity
    perplexity = evaluate.load("perplexity", module_type="metric", cache_dir="evaluation_metrics")

    perplexity_score = perplexity.compute(model_id='gpt2',
                                add_start_token=False,
                                predictions=predictions, 
                                batch_size =2)
    
    print(f"Perplexity: {perplexity_score['mean_perplexity']}")

    # Meteor
    meteor = evaluate.load('meteor', cache_dir="evaluation_metrics")
    meteor_score = meteor.compute(predictions=predictions, references=long_answers)
    print(f'Meteor: {meteor_score}')

    # Accuracy
    accuracy = acc_calc_final(predictions=predictions, references=final_decisions)
    print(f"Accuracy: {accuracy}")

    if args.save_dir:
        if args.verbose:
            print("Saving Results")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        filename = os.path.join(args.save_dir , f"{time.strftime('%Y%m%d-%H%M%S')}_{args.retrieval_type}_{args.approach}-{args.prompt_type}.json")
        if args.verbose:
            print(f"Saving Results to {filename}")
        
        data = {"model": model_name,
                # "prompt": prompt_templates[args.approach][args.prompt_type],
                'results': results, 'metrics': {'BertScore': bert_score, 'BLEU': bleu_score, "Accuracy": accuracy,  "Perplexity":perplexity_score['mean_perplexity'], "Meteor": meteor_score['meteor']},  #
                "ground_truth": {"long_answers": long_answers, "final_decisions": final_decisions}}

        json_data = json.dumps(data, indent=2)

        with open(filename, 'w') as json_file:
            json_file.write(json_data)
        if args.verbose:
            print("Results Saved")




if __name__ == '__main__':
    main()