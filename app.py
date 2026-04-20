"""
app.py — Brain Tumor Detection inference server.

Serves a FastAPI backend with:
  POST /api/predict   — upload MRI image, get detection + classification results
  GET  /api/status    — model loading status
  GET  /              — serves the frontend SPA

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure working directory is always the project root
os.chdir(Path(__file__).parent)

import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.data.transforms import get_transforms
from src.models.backbone import build_backbone
from src.models.detection_head import DetectionHead
from src.models.classification_head import ClassificationHead
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Model assembly helpers (mirrors training scripts without local classes)
# ---------------------------------------------------------------------------

class _DetectionModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class _ClassificationModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# App & model state
# ---------------------------------------------------------------------------

logger = get_logger("app")
app = FastAPI(title="Brain Tumor Detection API", version="1.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_transforms("test", image_size=224)

_state: Dict = {
    "phase1_model": None,
    "phase2_model": None,
    "phase1_loaded": False,
    "phase2_loaded": False,
    "phase1_error": None,
    "phase2_error": None,
    "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"],
    "detection_threshold": 0.5,
}


def _load_phase1(cfg: dict) -> None:
    ckpt = cfg["output"]["checkpoint_dir"] + "/phase1_best.pth"
    if not Path(ckpt).exists():
        _state["phase1_error"] = f"Checkpoint not found: {ckpt}"
        logger.warning(_state["phase1_error"])
        return
    try:
        backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
        head = DetectionHead(backbone.out_features, cfg["model"]["dropout"])
        model = _DetectionModel(backbone, head)
        load_checkpoint(model, ckpt, device=DEVICE)
        model.to(DEVICE).eval()
        _state["phase1_model"] = model
        _state["phase1_loaded"] = True
        _state["detection_threshold"] = cfg["evaluation"]["threshold"]
        logger.info("Phase 1 model loaded.")
    except Exception as e:
        _state["phase1_error"] = str(e)
        logger.error(f"Phase 1 load failed: {e}")


def _load_phase2(cfg: dict) -> None:
    ckpt = cfg["output"]["checkpoint_dir"] + "/phase2_best.pth"
    if not Path(ckpt).exists():
        _state["phase2_error"] = f"Checkpoint not found: {ckpt}"
        logger.warning(_state["phase2_error"])
        return
    try:
        backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
        head = ClassificationHead(backbone.out_features, cfg["num_classes"], cfg["model"]["dropout"])
        model = _ClassificationModel(backbone, head)
        load_checkpoint(model, ckpt, device=DEVICE)
        model.to(DEVICE).eval()
        _state["phase2_model"] = model
        _state["phase2_loaded"] = True
        _state["class_names"] = cfg["class_names"]
        logger.info("Phase 2 model loaded.")
    except Exception as e:
        _state["phase2_error"] = str(e)
        logger.error(f"Phase 2 load failed: {e}")


@app.on_event("startup")
def startup_event() -> None:
    logger.info(f"Starting inference server on device: {DEVICE}")
    try:
        with open("config/phase1_config.yaml") as f:
            cfg1 = yaml.safe_load(f)
        _load_phase1(cfg1)
    except Exception as e:
        _state["phase1_error"] = str(e)

    try:
        with open("config/phase2_config.yaml") as f:
            cfg2 = yaml.safe_load(f)
        _load_phase2(cfg2)
    except Exception as e:
        _state["phase2_error"] = str(e)


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_detection(image: Image.Image) -> Dict:
    model = _state["phase1_model"]
    threshold = _state["detection_threshold"]

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    logit = model(tensor)
    prob = torch.sigmoid(logit).squeeze().item()
    detected = prob >= threshold

    return {
        "detected": bool(detected),
        "confidence": round(float(prob) * 100, 1),
    }


@torch.no_grad()
def _run_classification(image: Image.Image) -> Dict:
    model = _state["phase2_model"]
    class_names = _state["class_names"]

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    top_idx = int(torch.tensor(probs).argmax().item())
    class_probs = {name: round(p * 100, 1) for name, p in zip(class_names, probs)}

    return {
        "predicted_class": class_names[top_idx],
        "confidence": round(probs[top_idx] * 100, 1),
        "class_probabilities": class_probs,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status() -> JSONResponse:
    return JSONResponse({
        "device": str(DEVICE),
        "phase1": {
            "loaded": _state["phase1_loaded"],
            "error": _state["phase1_error"],
        },
        "phase2": {
            "loaded": _state["phase2_loaded"],
            "error": _state["phase2_error"],
        },
        "class_names": _state["class_names"],
    })


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if not _state["phase1_loaded"] and not _state["phase2_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Train Phase 1 and/or Phase 2 first, then restart the server.",
        )

    # Load and preprocess image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid MRI scan.")

    result: Dict = {}

    # Phase 1 — detection
    if _state["phase1_loaded"]:
        detection = _run_detection(image)
        result["detection"] = detection
    else:
        result["detection"] = None
        result["detection_error"] = _state["phase1_error"]

    # Phase 2 — classification (always run; useful even when Phase 1 not available)
    if _state["phase2_loaded"]:
        classification = _run_classification(image)
        result["classification"] = classification
    else:
        result["classification"] = None
        result["classification_error"] = _state["phase2_error"]

    # Derived summary
    if result.get("detection") and result.get("classification"):
        det = result["detection"]
        cls = result["classification"]
        result["summary"] = {
            "tumor_present": det["detected"],
            "detection_confidence": det["confidence"],
            "tumor_type": cls["predicted_class"] if det["detected"] else "none",
            "type_confidence": cls["confidence"] if det["detected"] else None,
        }
    elif result.get("detection"):
        det = result["detection"]
        result["summary"] = {
            "tumor_present": det["detected"],
            "detection_confidence": det["confidence"],
            "tumor_type": None,
            "type_confidence": None,
        }
    elif result.get("classification"):
        cls = result["classification"]
        detected = cls["predicted_class"] != "no_tumor"
        result["summary"] = {
            "tumor_present": detected,
            "detection_confidence": cls["confidence"],
            "tumor_type": cls["predicted_class"],
            "type_confidence": cls["confidence"],
        }

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse("frontend/index.html")
