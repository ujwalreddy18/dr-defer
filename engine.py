"""Model + analysis logic for DR-Defer (no UI code here).

For research use only - NOT a medical device.
Pipeline matches the training/evaluation notebooks exactly:
  crop black border -> pad to square -> resize 512 -> JPEG q92 -> ImageNet normalise
  -> 3 x EfficientNet-B0 (seeds 0,1,2) -> per-model softmax(logits / T_i) -> mean
  -> ensemble temperature -> softmax(log(p) / TE)
Deferral rule: defer when (1 - max probability) > TAU.
"""
import base64
import io
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 512
SEEDS = [0, 1, 2]
TEMPS = [2.073338462176734, 1.5841914679687727, 1.8309191355951568]  # per model, fitted on EyePACS validation
TE = 0.8318839145884431                                              # extra temperature for the ensemble
TAU = 0.5013038516044617                                             # defer when confidence < 0.4987
STAGES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

device = "cuda" if torch.cuda.is_available() else "cpu"
models = []
for s in SEEDS:
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    m.load_state_dict(torch.load(os.path.join(HERE, f"best_512_s{s}.pt"), map_location="cpu"))
    models.append(m.to(device).eval())

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _crop_resize(arr, size=SIZE):
    mask = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) > 10
    if mask.mean() > 0.1:
        ys, xs = np.where(mask)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _preprocess(pil_img):
    arr = _crop_resize(np.array(pil_img.convert("RGB")))
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 92])
    arr = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(((arr.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).copy())
    return Image.fromarray(arr), x.unsqueeze(0)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _ensemble_probs(x):
    ps = []
    with torch.no_grad():
        for m, t in zip(models, TEMPS):
            ps.append(_softmax(m(x.to(device))[0].float().cpu().numpy() / t))
    p = np.mean(ps, 0)
    return _softmax(np.log(p + 1e-12) / TE)


def _gradcam(x, cls):
    """Average Grad-CAM of the three models for the ensemble's predicted class."""
    cams = []
    for m in models:
        with torch.enable_grad():
            feats = m.forward_features(x.to(device))
            feats.retain_grad()
            logits = m.forward_head(feats)
            m.zero_grad()
            logits[0, cls].backward()
            w = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * feats).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cams.append(cam / (cam.max() + 1e-8))
    cam = np.mean(cams, 0)
    return cam / (cam.max() + 1e-8)


def _overlay(pil, cam):
    retina = np.asarray(pil.convert("L")) > 10
    cam = cam * retina
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.asarray(pil).astype(np.float32)
    a = (0.45 * (cam > 0.05))[..., None]
    return Image.fromarray((base * (1 - a) + heat * a).astype(np.uint8))


def _data_url(pil, quality=85):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def analyse(pil_img):
    """Return a JSON-serialisable dict describing the prediction."""
    pil, x = _preprocess(pil_img)
    p = _ensemble_probs(x)
    pred, conf = int(p.argmax()), float(p.max())
    defer = (1 - conf) > TAU

    rgb = np.asarray(pil).astype(np.float32)
    on_retina = rgb.mean(2) > 10
    retina_share = float(on_retina.mean())
    red_ratio = (float(rgb[..., 0][on_retina].mean() / (rgb[..., 2][on_retina].mean() + 1))
                 if on_retina.any() else 0.0)
    not_fundus = bool(retina_share < 0.35 or red_ratio < 1.5)

    heat = _overlay(pil, _gradcam(x, pred)) if pred != 0 else None
    return {
        "stage": pred,
        "stage_name": STAGES[pred],
        "confidence": conf,
        "defer": bool(defer),
        "referable": bool(pred >= 2),
        "threshold": 1 - TAU,
        "probabilities": [float(v) for v in p],
        "stages": STAGES,
        "not_fundus": not_fundus,
        "image": _data_url(pil),
        "heatmap": _data_url(heat) if heat is not None else None,
    }
