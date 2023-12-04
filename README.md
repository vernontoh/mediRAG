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

### LLM Experiments base
```bash
python llm_experiments_best_ret.py --hf_token <HF Token>  --n_samples 100 --approach vanilla --prompt_type base --save_dir eval
```
you can run the experiments for other approaches {"vanilla" , "citations" , "retrieval", "stepback"} by replacing the argument to the --approach and other prompt {"cot" , "base" , "one-shot"} by replacing the --prompt_type argument.

### LLM Experiment with Retrieval and Using Chain-of-Thought prompting
```bash
python llm_experiments_best_ret.py --hf_token <HF Token>  --n_samples 100 --approach retrieval --prompt_type cot --save_dir eval
```