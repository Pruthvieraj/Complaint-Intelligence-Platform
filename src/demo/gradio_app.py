"""
Phase 6 / Phase 7 — minimal demo UI. A recruiter pastes a complaint,
gets a live prediction, confidence, and per-request latency, powered
by the exact same ComplaintClassifier the FastAPI service uses (see
src/serving/inference.py) — so the demo can never show something the
API wouldn't.
"""
import gradio as gr

from src.serving.inference import ComplaintClassifier

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ComplaintClassifier()
    return _classifier


EXAMPLES = [
    "XXXX has been calling me multiple times a day about a debt of $499.00 that I do not believe is mine. "
    "I asked for debt validation and never received anything in writing.",
    "I noticed a charge of $250.00 on my XXXX card statement that I did not authorize. I called to dispute it "
    "and was told it would take 5 business days to investigate.",
    "XXXX repossessed my vehicle despite my payment clearing two days earlier.",
    "My credit report shows a collections account from XXXX that was already paid off. This is affecting my "
    "ability to get approved for a loan.",
]


def predict(text: str):
    if not text or not text.strip():
        return "—", "", ""
    clf = _get_classifier()
    result = clf.predict(text, top_k=3)
    top_k_str = "\n".join(f"{r['label']}: {r['probability']:.1%}" for r in result["top_k"])
    return (
        f"{result['predicted_category']}  ({result['confidence']:.1%} confidence)",
        top_k_str,
        f"{result['latency_ms']:.1f} ms",
    )


with gr.Blocks(title="Complaint Intelligence Platform") as demo:
    gr.Markdown(
        "# Complaint Intelligence Platform\n"
        "Paste a consumer complaint narrative and get a live product/issue category "
        "prediction from the ONNX-quantized, promotion-gated model."
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(lines=6, label="Complaint narrative")
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=text_in)
        with gr.Column():
            pred_out = gr.Textbox(label="Predicted category")
            topk_out = gr.Textbox(label="Top-3 probabilities", lines=3)
            latency_out = gr.Textbox(label="Inference latency")

    btn.click(predict, inputs=text_in, outputs=[pred_out, topk_out, latency_out])
    text_in.submit(predict, inputs=text_in, outputs=[pred_out, topk_out, latency_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
