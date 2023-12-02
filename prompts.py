





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

MEDITRON_PROMPT = """
<|im_start|>system
Answer the users question based only on the following context:<|im_end|>
<|im_start|>
<context>
{context}
</context>

question
{question}<|im_end|>
<|im_start|>answer  
"""

BASE_PROMPT = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}  
"""
# What subproblems must be solved before answering the inquiry?

COT_PROMPT = """Think Step by Step and Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}  
"""

FEW_SHOT_PROMPT = """Answer the users question based only on the following context:

<example>
Question:
Does neurobehavioral disinhibition predict initiation of substance use in children with prenatal cocaine exposure?
Answer:
Yes, Prenatal drug exposure appears to be a risk pathway to ND, which by 8/9 years portends substance use initiation.
</example>

<context>
{context}
</context>

Question: {question}  
"""

# TOT_PROMPT = """Answer the users question based only on the following context. What subproblems must be solved before answering the inquiry?:

# <context>
# {context}
# </context>

# Question: {question}
# Answer in the format:
# Step| Subproblem | Answer to Subproblem  

# Final Answer:
# """