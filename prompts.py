





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
