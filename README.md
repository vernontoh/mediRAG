# mediRAG


## Install required packages
```bash
pip install -r requirements.txt
```

## Retrieval Evaluations

### Experiment 1 (Dense Embedding)
```bash
python retrieval_experiments.py --task single
```

### Experiment 2 (Dense Ensemble)
```bash
python retrieval_experiments.py --task multi_dense
```

### Experiment 3 (Hybrid Ensemble)
```bash
python retrieval_experiments.py --task hybrid
```

### Experiment 4 (Reranking)
```bash
python retrieval_experiments.py --task rerank 
```

### Experiment 5 (Query Expansion)
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