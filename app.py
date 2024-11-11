import gradio as gr
from app.chatbot import Chatbot

def create_app():
    chatbot = Chatbot()

    def respond(message, history):
        if message.strip() == "":
            return "", history
        bot_message = chatbot.generate_response(message)
        history.append((message, bot_message))
        return "", history

    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🤖 AI Chatbot
        Start chatting below! The model might take a few moments to respond.
        """)
        
        chatbot_component = gr.Chatbot(
            value=[],
            height=400,
            bubble_full_width=False,
            show_label=True,
            label="Chat History"
        )
        
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            container=False
        )
        
        with gr.Row():
            clear = gr.ClearButton([msg, chatbot_component], value="Clear Chat")
            submit = gr.Button("Send", variant="primary")

        msg.submit(respond, [msg, chatbot_component], [msg, chatbot_component])
        submit.click(respond, [msg, chatbot_component], [msg, chatbot_component])

    return demo

# Create the demo application
demo = create_app()

# This is important for Gradio/HuggingFace Spaces deployment
if __name__ == "__main__":
    demo.launch()
else:
    app = demo.app
