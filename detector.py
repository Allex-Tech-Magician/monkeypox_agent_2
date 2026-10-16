"""Vision part of the agent: build the three models, load your trained weights,
check and prepare an image, predict, and vote.

No Streamlit and no LLM code lives here, so this file can be tested on its own.
"""
import io
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ORDER MATTERS: index 0 = Monkeypox, index 1 = Others (the ImageFolder order from training).
CLASS_NAMES = ("Monkeypox", "Others")
MPOX_IDX = CLASS_NAMES.index("Monkeypox")
OTHERS_IDX = CLASS_NAMES.index("Others")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_SIDE = 32


# ---------------------------------------------------------------- architectures
class CustomCNN(nn.Module):
    """3 conv blocks + 3 linear layers. Needs 224x224 input (128 * 28 * 28)."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class ResNeXtModel(nn.Module):
    """torchvision ResNeXt-50 (32x4d) with the custom head used in training."""

    def __init__(self, num_classes):
        super().__init__()
        self.resnext = tv_models.resnext50_32x4d(weights=None)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.resnext(x)


def build_seresnext50(num_classes):
    """Real SE-ResNeXt-50 from timm. Weights come from your checkpoint (no download)."""
    return timm.create_model("seresnext50_32x4d", pretrained=False, num_classes=num_classes)


# display name -> (checkpoint file, function that builds the empty architecture)
MODEL_SPECS = {
    "Custom CNN": ("cnn_model.pth", CustomCNN),
    "ResNeXt-50": ("resnext_model.pth", ResNeXtModel),
    "SE-ResNeXt-50": ("seresnext_model.pth", build_seresnext50),
}
MODEL_FILES = {name: spec[0] for name, spec in MODEL_SPECS.items()}


# ---------------------------------------------------------------- loading
def _load_one(name, device):
    filename, builder = MODEL_SPECS[name]
    path = MODELS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"{filename} was not found in the models folder")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):  # accept a state_dict wrapped inside another dict
        for key in ("model_state_dict", "state_dict"):
            if isinstance(raw.get(key), dict):
                raw = raw[key]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("not a state_dict; re-save with torch.save(model.state_dict(), path)")

    state = {k.removeprefix("module."): v for k, v in raw.items()}  # DataParallel prefix
    model = builder(num_classes=len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)  # a wrong architecture fails loudly
    return model.to(device).eval()


def load_models():
    """Load all three models. Returns (models, errors, device); never crashes the app."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, errors = {}, {}
    for name in MODEL_SPECS:
        try:
            models[name] = _load_one(name, device)
        except Exception as exc:  # report each problem instead of stopping
            errors[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return models, errors, device


# ---------------------------------------------------------------- image + prediction
class ImageError(Exception):
    """The upload cannot be used as an image. The message is safe to show to the user."""


def check_image(file_bytes):
    """Validate the upload and return an RGB PIL image."""
    if not file_bytes:
        raise ImageError("The file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ImageError("The image is larger than 10 MB.")
    try:
        Image.open(io.BytesIO(file_bytes)).verify()  # integrity check
        image = Image.open(io.BytesIO(file_bytes))   # verify() leaves it unusable, so reopen
        image_format = image.format
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageError("That file is not a readable image.") from exc
    if image_format not in ("JPEG", "PNG"):
        raise ImageError("Please use a JPG or PNG image.")
    image = ImageOps.exif_transpose(image)  # phone photos may be rotated
    if min(image.size) < MIN_SIDE:
        raise ImageError("The image is too small. Please upload a clearer photo.")
    return image.convert("RGB")


# The same eval-time pipeline as in the training notebooks.
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(models, image, device):
    """Run every model on one image. Returns a list of dicts, one per model."""
    batch = _TRANSFORM(image).unsqueeze(0).to(device)  # shape [1, 3, 224, 224]
    predictions = []
    with torch.inference_mode():
        for name, model in models.items():
            probs = torch.softmax(model(batch), dim=1)[0].cpu().tolist()
            mpox, others = float(probs[MPOX_IDX]), float(probs[OTHERS_IDX])
            label = CLASS_NAMES[MPOX_IDX] if mpox >= others else CLASS_NAMES[OTHERS_IDX]
            predictions.append({"model": name, "label": label, "monkeypox": mpox, "others": others})
    return predictions


def vote(predictions):
    """Majority vote. A tie is reported as 'Tie', never decided silently."""
    counts = {name: 0 for name in CLASS_NAMES}
    for p in predictions:
        counts[p["label"]] += 1
    top = max(counts.values())
    winners = [name for name, n in counts.items() if n == top]
    final = winners[0] if len(winners) == 1 else "Tie"
    agreement = f"{top}/{len(predictions)}" + (" (tie)" if final == "Tie" else "")
    average = sum(p["monkeypox"] for p in predictions) / len(predictions)
    return {"final": final, "agreement": agreement, "votes": counts, "avg_monkeypox": average}


# (lean, agreement) -> headline shown to the user
HEADLINES = {
    ("monkeypox", "high"): "Looks like monkeypox",
    ("monkeypox", "moderate"): "Probably monkeypox",
    ("monkeypox", "low"): "Leaning monkeypox, but unsure",
    ("not_monkeypox", "high"): "Doesn't look like monkeypox",
    ("not_monkeypox", "moderate"): "Probably not monkeypox",
    ("not_monkeypox", "low"): "Leaning not monkeypox, but unsure",
}


def assess(predictions, summary):
    """A plain-language verdict from the votes and how sure each model is.

    `agreement` describes how well the models agree with each other. It is NOT the chance
    that the person has the disease.
    """
    if summary["final"] == "Tie":
        return {
            "lean": "unclear",
            "agreement": "low",
            "headline": "Can't tell: the models are split",
            "note": "The models split evenly, so there is no majority.",
        }
    final = summary["final"]
    own = {p["model"]: (p["monkeypox"] if p["label"] == "Monkeypox" else p["others"]) for p in predictions}
    winners = [p for p in predictions if p["label"] == final]
    dissent = [p for p in predictions if p["label"] != final]
    strong_dissent = any(own[p["model"]] >= 0.8 for p in dissent)
    if not dissent and min(own[p["model"]] for p in winners) >= 0.8:
        agreement = "high"
    elif not dissent or not strong_dissent:
        agreement = "moderate"
    else:
        agreement = "low"
    lean = "monkeypox" if final == "Monkeypox" else "not_monkeypox"
    if dissent:
        names = ", ".join(p["model"] for p in dissent)
        note = f"{names} disagreed {'strongly' if strong_dissent else 'only weakly'}."
    else:
        note = "All models agree."
    return {"lean": lean, "agreement": agreement, "headline": HEADLINES[(lean, agreement)], "note": note}


def analyze(models, file_bytes, device):
    """Image bytes in, plain results dictionary out (this is what the agent talks about)."""
    image = check_image(file_bytes)
    predictions = predict_image(models, image, device)
    summary = vote(predictions)
    return {
        "final": summary["final"],
        "agreement": summary["agreement"],
        "votes": summary["votes"],
        "avg_monkeypox_pct": round(summary["avg_monkeypox"] * 100, 2),
        "verdict": assess(predictions, summary),
        "models": [
            {
                "name": p["model"],
                "label": p["label"],
                "monkeypox_pct": round(p["monkeypox"] * 100, 2),
                "others_pct": round(p["others"] * 100, 2),
            }
            for p in predictions
        ],
    }
