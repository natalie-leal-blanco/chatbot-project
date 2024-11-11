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

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
        css="#chatbot {height: 400px; overflow-y: auto;}"
    ) as demo:
        gr.Markdown("""
        # 🤖 AI Chatbot
        Start chatting below! The model might take a few moments to respond.
        """)
        
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            height=400
        )
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            container=False
        )
        with gr.Row():
            clear = gr.ClearButton([msg, chatbot], value="Clear Chat")
            submit = gr.Button("Send", variant="primary")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])

    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(debug=True)
