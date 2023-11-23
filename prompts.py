





PROMPT_TEMPLATE_QA = """### [INST] 
Instruction: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{context}

### QUESTION:
{question} 

[/INST]"""


PROMPT_TEMPLATE_QA_TRUTHFUL = """### [INST] 
Instruction: Use the following pieces of context to answer the question at the end.

{context}

### QUESTION:
{question} 

Give a detailed answer. If the context does not provide the answer, say the answer cannot be found from the 
given context. Always start with OK....
[/INST]"""


PROMPT_TEMPLATE_QA_EXPLAINER = """### [INST] 
Instruction: Use the following pieces of context to answer the question at the end.

{context}

### QUESTION:
{question} 

Give a detailed answer with the help of the provided piece of information.
[/INST]"""

PROMPT_TEMPLATE_QA_ANSWER = """### [INST] 
Instruction: Use the following pieces of context to answer the question at the end.

{context}

### QUESTION:
{question} 

Give a detailed answer with the help of the provided piece of information. Always end with "The Answer is [Yes,No,Maybe]" to answer the question based on the provided information.
[/INST]""" ## Acc = 0.3 for 10 samples BERTScore: {'precision': 0.8313113331794739} BLEU: 0.031105619821945752

PROMPT_TEMPLATE_QA_ANSWER_START = """### [INST] 
Instruction: Use the following pieces of context to answer the question at the end.

{context}

### QUESTION:
{question} 

Give a detailed answer with the help of the provided piece of information. Always start with "The Answer is [Yes,No,Maybe]" to answer the question based on the provided information.
[/INST]""" ## Acc = 0.6 for 10 samples BERTScore: {'precision': 0.821868222951889}, BLEU: 0.024074726951251374