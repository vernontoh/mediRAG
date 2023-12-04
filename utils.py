import wikipedia
import numpy as np
from pymed import PubMed
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def preprocess(dataset):
    page_content_column = "CONTEXTS"
    for split in dataset.keys():
        for contexts in dataset[split][page_content_column]:
            for sentence in contexts:
                yield Document(page_content=sentence)

def create_question2chunk(val_questions, val_contexts, chunk_size=1024, chunk_overlap=128):
    question2chunk = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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


def rerank_topk(reranker, question, documents):
    all_docs_ls = []
    for document in documents:
        doc_content = document.page_content
        qs_doc_ls = [question, doc_content]
        all_docs_ls.append(qs_doc_ls)
    scores = reranker.compute_score(all_docs_ls)
    zipped_lists = list(zip(scores, all_docs_ls))
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    sorted_values, sorted_original = zip(*sorted_lists)
    result = [nested_list[1] for nested_list in sorted_original]
    result_new = [Document(page_content=content) for content in result]
    return result_new


def query_expansion(llm, question, type):
    if type == "prompt":
        template = """Explain the biomedical terms and concepts in the following question:\n{question}\n\nAnswer: Let’s think step by step."""

        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)
        explanation = llm_chain.run(question)
        return question + explanation
    
    elif type == "wikipedia":
        template = """Question: Extract the most important biomedical term in the following question: {question}.

        Answer: """

        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)
        extraction = llm_chain.run(question)

        try:
            summary = wikipedia.summary(extraction, sentences=1)
            res = question+summary
        except:
            res = question
        return res
    
    elif type == "pubmed":
        pubmed = PubMed(tool="MyTool", email="leepeixuan133@gmail.com")
        expanded = question
        try:
            results = pubmed.query(question, max_results=2)
            titles = ""
            for article in results:
                titles += " " + article.title 
            expanded += titles
        except:
            expanded += ""
        return expanded




def preprocess(dataset):
    for split in dataset.keys():
        for contexts in dataset[split]["CONTEXTS"]:
            for sentence in contexts:
                yield Document(page_content=sentence)

# Parser to remove the `**`
def _parse(text):
    return text.strip("**")

def find_citations(predictions, db):
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