# mediRAG


## Install required packages
```bash
pip install -r requirements.txt
```

## Retrieval Evaluations

### Experiment 1 (Chunk Configuration)
```bash
python retrieval_experiments.py --task chunk
```

### Experiment 2 (Dense Embedding)
```bash
python retrieval_experiments.py --task single
```

### Experiment 3 (Dense Ensemble)
```bash
python retrieval_experiments.py --task multi_dense
```

### Experiment 4 (Hybrid Ensemble)
```bash
python retrieval_experiments.py --task hybrid
```

### Experiment 5 (Reranking)
```bash
python retrieval_experiments.py --task rerank 
```

### Experiment 6 (Query Expansion)
```bash
python retrieval_experiments.py --task query_expand --hf_token <HF Token>
```



## LLM Evaluations

### Experiment with no retrieval

```bash
python llm_experiments.py \
--approach vanilla \
--prompt_type base \
```
**NOTE**: You can run the experiments with other configurations.  
```python
--approach: ["vanilla", "citations"]  
--prompt_type: [ "base", "cot", "one-shot"]  
```

### Experiment with retrieval

```bash
python llm_experiments.py \
--retrieval_type base \
--approach retrieval \
--prompt_type base \
--hf_token <HF Token> \
```
**NOTE**: You can run the experiments with other configurations.  
```python
--retrieval_type: ["base", "best"]
--approach: ["retrieval", "stepback"]  
--prompt_type: ["base", "cot", "one-shot"]  
```


