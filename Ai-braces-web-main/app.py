import os
import uuid
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from utils.unet_inference  import load_unet,  run_unet
from utils.yolo_inference  import load_yolo,  run_yolo

# ─────────────────────────────────────────────
# Paths  – edit these two lines if needed
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UNET_PATH  = os.path.join(BASE_DIR, "..", "root_model2.pth")
YOLO_PATH  = os.path.join(BASE_DIR, "..", "Bracket detection roboflow",
                          "Bracket detection roboflow", "best.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB max upload


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────────
# Load models ONCE at startup
# ─────────────────────────────────────────────
print("Loading models…")
load_unet(UNET_PATH)
load_yolo(YOLO_PATH)
print("Models ready.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save upload
    uid = uuid.uuid4().hex[:8]
    filename = uid + "_" + secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    mode = request.form.get("mode", "unet")

    try:
        def save(name, img):
            path = os.path.join(OUTPUT_DIR, f"{uid}_{name}.jpg")
            cv2.imwrite(path, img)
            return f"outputs/{uid}_{name}.jpg"

        # ── Mode: UNet only ───────────────────
        if mode == "unet":
            unet_result  = run_unet(upload_path)
            axis_url     = save("axis",     unet_result["axis_img"])
            brackets_url = save("brackets", unet_result["bracket_img"])
            mask_url     = save("mask",     unet_result["mask"] * 255)
            return jsonify({
                "mode":           "unet",
                "axis":           axis_url,
                "brackets":       brackets_url,
                "mask":           mask_url,
                "teeth_detected": len(unet_result["bracket_points"]),
                "jaw":            "Lower" if unet_result["lower_jaw"] else "Upper",
            })

        # ── Mode: YOLO only ───────────────────
        elif mode == "yolo":
            yolo_result = run_yolo(upload_path)
            yolo_url    = save("yolo", yolo_result["plotted_img"])
            confs       = [round(b["conf"], 3) for b in yolo_result["boxes"]]
            return jsonify({
                "mode":           "yolo",
                "yolo":           yolo_url,
                "brackets_found": len(yolo_result["boxes"]),
                "confidences":    confs,
            })

        else:
            return jsonify({"error": f"Unknown mode: {mode}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
