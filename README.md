# mediRAG


## Install required packages
```bash
pip install -r requirements.txt
```

## Run Evaluations

### Experiment 1 (Comparision between BGE Embedding and PubMed Bert Embedding)
```bash
python retrieval_experiments.py --task single
```

### Experiment 2 (Using 2 Dense embeddings and tuning the weightage)
```bash
python retrieval_experiments.py --task multi_dense
```

### Experiment 3 (Using BM25 and Dense Embeddings and tuning the weightage)
```bash
python retrieval_experiments.py --task hybrid
```

### Experiment 4 (Using 2 Dense Embeddings and tuning the topk for reranking)
```bash
python retrieval_experiments.py --task rerank 
```

### Experiment 5 (Using 2 Dense Embeddings and refining prompts for Query Expansion)
```bash
python retrieval_experiments.py --task query_expand --hf_token <HF Token>
```