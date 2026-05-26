import math
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np

DEVICE = torch.device("cpu")
CROWN_OFFSET = 0.12

_model = None


def load_unet(model_path: str):
    """Load UNet model once at startup. Call this from app.py."""
    global _model
    _model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse",
    )
    _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _model = _model.to(DEVICE)
    _model.eval()
    print(f"[UNet] Model loaded from {model_path}")


def run_unet(image_path: str):
    """
    Run UNet inference on a dental OPG image.

    Args:
        image_path: Path to input image (grayscale OPG).

    Returns:
        dict with keys:
            mask          – binary mask (H x W uint8)
            mask_rgb      – mask as RGB image for display
            axis_img      – original image with tooth axes + bracket points drawn
            bracket_points – list of (x, y) ideal bracket coords (one per tooth)
    """
    if _model is None:
        raise RuntimeError("UNet model not loaded. Call load_unet() first.")

    # --- Pre-process ---
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original = cv2.resize(img_gray, (512, 512))

    inp = original / 255.0
    inp_tensor = torch.tensor(inp).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        pred = torch.sigmoid(_model(inp_tensor))

    pred_mask = (pred.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

    # --- Jaw orientation ---
    img_cy = original.shape[0] / 2
    ys_all, _ = np.where(pred_mask)
    if len(ys_all) == 0:
        raise RuntimeError("Empty mask — no teeth detected in image.")
    lower_jaw = ys_all.mean() < img_cy

    # --- Build output canvases (two separate images) ---
    base = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    axis_img    = base.copy()   # axis lines + apex only
    bracket_img = base.copy()   # bracket points only

    bracket_points = []

    num_labels, labels = cv2.connectedComponents(pred_mask)

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)

        if len(xs) < 50:
            continue

        if lower_jaw:
            crown_idx = np.argmax(ys)
            apex_idx  = np.argmin(ys)
        else:
            crown_idx = np.argmin(ys)
            apex_idx  = np.argmax(ys)

        crown_tip = (int(xs[crown_idx]), int(ys[crown_idx]))
        apex_pt   = (int(xs[apex_idx]),  int(ys[apex_idx]))

        bracket_pt = (
            int(crown_tip[0] + CROWN_OFFSET * (apex_pt[0] - crown_tip[0])),
            int(crown_tip[1] + CROWN_OFFSET * (apex_pt[1] - crown_tip[1])),
        )
        bracket_points.append(bracket_pt)

        # axis image: blue axis + red apex + green bracket (matches final.py)
        cv2.line(axis_img,   crown_tip, apex_pt, (255, 0, 0), 2)
        cv2.circle(axis_img, apex_pt,   4, (0, 0, 255), -1)
        cv2.circle(axis_img, bracket_pt, 4, (0, 255, 0), -1)

        # bracket image: cyan/yellow dot only (matches final.py opg_dots)
        cv2.circle(bracket_img, bracket_pt, 5, (255, 255, 0), -1)

    return {
        "axis_img":       axis_img,
        "bracket_img":    bracket_img,
        "mask":           pred_mask,
        "bracket_points": bracket_points,
        "lower_jaw":      lower_jaw,
    }
