import gradio as gr
from utils import *
from final_pipeline import QA

def QA_pipeline(question):
    try:
        Answer = QA(question)
    except Exception as error:
        Answer = error
    return Answer

examples = ["Does neurobehavioral disinhibition predict initiation of substance use in children with prenatal cocaine exposure?",\
            "Does the CTCF insulator protein form an unusual DNA structure?",\
            'Is exercise Medicine™ : A pilot study linking primary care with community physical activity support?',\
                ]

with gr.Blocks(theme = gr.themes.Soft()) as demo:
    gr.Markdown(
    """
    # MediRAG
    Chatbot for quick and verified answers to biomedical questions?
    """)
    chatbot = gr.Chatbot(label = 'MediRag')
    msg = gr.Textbox(label = 'Enter Question')
    clear = gr.ClearButton([msg, chatbot])

    # gr.Markdown("## Sample Biomedical Questions")
    gr.Examples(
        examples,
        [msg],label = 'Sample Biomedical Questions'
    )

    def respond(message, chat_history):
        bot_message = QA_pipeline(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch() 