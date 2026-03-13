from __future__ import annotations

import os

import gradio as gr

from .config import Settings
from .service import RAGSystem


def build_demo() -> gr.Blocks:
    settings = Settings.from_env()
    rag = RAGSystem(settings)

    def sync_index(force_rebuild: bool) -> tuple[str, str]:
        try:
            summary = rag.sync_documents(force=force_rebuild)
            return rag.format_status_markdown(summary), "Index refresh completed."
        except Exception as exc:
            return rag.format_status_markdown(), f"Index refresh failed: {exc}"

    def respond(
        message: str,
        history: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], str, str, str]:
        history = history or []
        if not message.strip():
            return history, rag.format_sources_markdown([]), rag.format_status_markdown(), ""

        try:
            answer, hits = rag.answer_question(message, history)
        except Exception as exc:
            answer = f"Request failed: {exc}"
            hits = []

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, rag.format_sources_markdown(hits), rag.format_status_markdown(), ""

    with gr.Blocks(title="PDF RAG Assistant") as demo:
        gr.Markdown(
            """
            # PDF RAG Assistant
            Drop PDFs into `data/pdfs/`, index them, then chat against the retrieved document context.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", type="messages", height=560)
                with gr.Row():
                    message_box = gr.Textbox(
                        label="Question",
                        placeholder="Ask something grounded in your PDFs...",
                        scale=8,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    clear_button = gr.Button("Clear Chat")
            with gr.Column(scale=2):
                status_box = gr.Markdown(rag.format_status_markdown())
                force_checkbox = gr.Checkbox(label="Force full rebuild", value=False)
                ingest_button = gr.Button("Refresh Index", variant="secondary")
                sync_result = gr.Markdown("Index status idle.")
                sources_box = gr.Markdown("## Sources\nNo sources used yet.")

        send_button.click(
            fn=respond,
            inputs=[message_box, chatbot],
            outputs=[chatbot, sources_box, status_box, message_box],
        )
        message_box.submit(
            fn=respond,
            inputs=[message_box, chatbot],
            outputs=[chatbot, sources_box, status_box, message_box],
        )
        clear_button.click(lambda: ([], "## Sources\nNo sources used yet."), outputs=[chatbot, sources_box])
        ingest_button.click(fn=sync_index, inputs=[force_checkbox], outputs=[status_box, sync_result])

    return demo


def main() -> None:
    demo = build_demo()
    launch_kwargs: dict[str, object] = {}

    server_name = os.getenv("GRADIO_SERVER_NAME", "").strip()
    server_port = os.getenv("GRADIO_SERVER_PORT", "").strip()

    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = int(server_port)

    try:
        demo.launch(**launch_kwargs)
    except OSError as exc:
        message = str(exc)
        if "Cannot find empty port" in message:
            raise SystemExit(
                "Gradio could not bind a local port. Set GRADIO_SERVER_PORT to a free port if "
                "another process is using the default range. In restricted environments, local "
                "port binding may be blocked entirely."
            ) from exc
        raise
