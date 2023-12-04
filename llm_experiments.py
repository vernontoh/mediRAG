import argparse
import os
import gc
import torch
import numpy as np
from prompts import *
import evaluate
import json
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='A script to evaluate the performance of different LLM prompts and retrieval query approaches')

    parser.add_argument('--prompt_type', choices=["cot" , "base" , "one-shot"], default="base")

    parser.add_argument('--approach', choices=["vanilla" , "citations" , "retrieval", "stepback"], default="vanilla")
    
    parser.add_argument('--n_samples' , type=int , default=100)

    parser.add_argument('--cuda_device' , type=int , default=0)

    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose mode')
    
    args = parser.parse_args()
    return args

def preprocess(dataset):
    for split in dataset.keys():
        for contexts in dataset[split]["CONTEXTS"]:
            for sentence in contexts:
                yield Document(page_content=sentence)

# Parser to remove the `**`
def _parse(text):
    return text.strip("**")

def find_citations(predictions):
    finalpred = []
    for output in predictions:
        output_with_citations = ""
        citations = ""
        citation_list = []

        for lines in output.split("\n"):
            lines = lines.strip()
            if len(lines.split(" ")) > 10:
                for line in lines.split("."):
                    line = line.strip()
                    docs_and_scores = db.similarity_search_with_score(line)[0]  # choosing top 1 relevant document
                    if docs_and_scores[1] < 0.5:  # returned distance score is L2 distance, a lower score is better
                        doc_content = docs_and_scores[0].page_content
                        if doc_content in citation_list:
                            idx = citation_list.index(doc_content)

                        else:
                            citation_list.append(doc_content)
                            idx = len(citation_list)
                            citations += f"[{idx}] {doc_content}\n"

                        output_with_citations += line + f" [{idx}]. "

        final_output_with_citations = output_with_citations + "\n\nCitations:\n" + citations
        finalpred.append(final_output_with_citations)
    return finalpred

def clean_cit(preds):
    new_pred = []
    for pred in preds:
        new_pred.append(pred.split("\n\nCitations:\n")[0])
    return new_pred

def acc_calc_final(predictions, references):
    acc = 0
    for i in range(len(predictions)):
        if references[i].lower() in predictions[i][:15].lower():
            acc += 1
    return acc / len(predictions)

if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    # set cuda device
    if args.verbose:
        print(f"Setting CUDA device to {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)  
    torch.cuda.current_device()

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
        device_map = "auto"
    )

    # Load data
    if args.verbose:
        print("Loading Dataset")
    dataset = load_dataset("bigbio/pubmed_qa", cache_dir="data")
    data = list(preprocess(dataset))

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

    if os.path.exists("faiss_index_pubmed"):
        db = FAISS.load_local("faiss_index_pubmed", embeddings)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        docs = text_splitter.split_documents(data)  # 676307

        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index_pubmed")

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

    # Getting Outputs
    if args.verbose:
        print("Generating Outputs")
    results = []
    for query in questions:    
        if args.verbose:
            print(f"Question: {query}")
        result = chain.invoke(query)
        if args.verbose:
            print(f"Generated Output: {result}")
        results.append(result)
    if args.approach == 'citations':
        if args.verbose:
            print(f"Generating Citations")
        results = find_citations(results)
    predictions = results
    if args.approach == 'citations':
        predicitons = clean_cit(predictions)

    # Evaluation
    del model, text_generation_pipeline, mistral_llm, chain, retriever, db
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
        filename = os.path.join(args.save_dir , f"{args.approach}-{args.prompt_type}-{args.n_samples}.json")
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






