import gradio as gr
from app.chatbot import Chatbot

def create_app():
    chatbot = Chatbot()

    def respond(message, chat_history):
        if message.strip() == "":
            return "", chat_history
        bot_message = chatbot.generate_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    demo = gr.Interface(
        fn=respond,
        inputs=[
            gr.Textbox(placeholder="Type your message here..."),
            gr.State([])
        ],
        outputs=[
            gr.Textbox(),
            gr.State()
        ],
        title="🤖 AI Chatbot",
        description="Start chatting below! The model might take a few moments to respond."
    )
    
    return demo

# Create the demo application
demo = create_app()

# This is important for Gradio/HuggingFace Spaces deployment
if __name__ == "__main__":
    demo.launch()
else:
    # This is the crucial part for HuggingFace Spaces
    app = demo.app
