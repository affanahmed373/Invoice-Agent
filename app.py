"""Gradio entrypoint for running Invoice Agent in Hugging Face Spaces."""

import gradio as gr

from invoice_agent import InvoiceAgent

agent = None


def answer(question: str) -> str:
    global agent
    if not question.strip():
        return "Please enter a question about the invoices."
    if agent is None:
        agent = InvoiceAgent()
    return agent.ask(question)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Invoice question", placeholder="Which vendor has the cheaper price for atta?"),
    outputs=gr.Markdown(label="Answer"),
    title="Invoice Agent",
    description="Ask questions about invoices indexed in Qdrant.",
)


if __name__ == "__main__":
    demo.launch()