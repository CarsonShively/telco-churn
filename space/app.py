from __future__ import annotations

import os
import gradio as gr

from telco_churn.serving.serve import ServingService

_svc: ServingService | None = None
_init_error: str | None = None


def init_service():
    global _svc, _init_error
    try:
        _svc = ServingService.start()
        _init_error = None

        prefix = _svc.fs.current_run_prefix()
        ids = _svc.fs.sample_entity_ids(limit=30)

        status = {
            "ready": True,
            "model_run_id": getattr(getattr(_svc, "artifact", None), "run_id", None),
            "current_prefix": prefix,
            "sample_id_count": len(ids),
            "error": None,
        }

        return status, gr.update(choices=ids, value=(ids[0] if ids else None))
    except Exception as e:
        _svc = None
        _init_error = f"{type(e).__name__}: {e}"
        status = {
            "ready": False,
            "model_run_id": None,
            "current_prefix": None,
            "sample_id_count": 0,
            "error": _init_error,
        }
        return status, gr.update(choices=[], value=None)


def _load_service() -> ServingService:
    if _svc is None:
        raise RuntimeError(_init_error or "Service not initialized. Click 'Init/Retry' first.")
    return _svc


def refresh_dropdown_ids(limit: int = 30):
    svc = _load_service()
    ids = svc.fs.sample_entity_ids(limit=limit)
    return gr.update(choices=ids, value=(ids[0] if ids else None))


def predict(customer_id: str) -> dict:
    customer_id = (customer_id or "").strip()
    if not customer_id:
        return {"ok": False, "error": "Please provide a customer_id", "result": None}

    try:
        svc = _load_service()
        res = svc.predict_customer(customer_id)
        return {"ok": True, "error": None, "result": res}
    except KeyError as e:
        return {"ok": False, "error": str(e), "result": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "result": None}


with gr.Blocks() as demo:
    gr.Markdown("# Telco Churn — Feature Store Demo")
    gr.Markdown(
        "Select a customer ID. Features are fetched from Redis and scored by the current champion model."

    )

    status = gr.JSON(label="Startup")

    with gr.Row():
        init_btn = gr.Button("Reload Service")

    with gr.Row():
        customer_id = gr.Dropdown(
            label="customer_id",
            choices=[],
            allow_custom_value=True,
            scale=3,
        )
        refresh_btn = gr.Button("Refresh 30 IDs", scale=1)
        btn = gr.Button("Predict", scale=1)

    out = gr.JSON(label="Response")

    demo.load(fn=init_service, inputs=[], outputs=[status, customer_id])
    init_btn.click(fn=init_service, inputs=[], outputs=[status, customer_id])
    refresh_btn.click(fn=refresh_dropdown_ids, inputs=[], outputs=[customer_id])
    btn.click(fn=predict, inputs=[customer_id], outputs=[out])

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

#uses logging and values sent on, NOT UI, what should be here?
# adds to config from this file?