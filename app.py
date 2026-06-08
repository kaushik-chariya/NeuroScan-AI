# app.py — NeuroScan-AI Flask Application (Full Featured)

import os
import sys
import uuid
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from io import BytesIO

import cv2
from flask import Flask, request, jsonify, render_template, send_file, abort
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from logger import logger
from exception import NeuroScanException
from src.pipeline.prediction_pipeline import PredictionPipeline

# ── App Init ──────────────────────────────────────────
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['UPLOAD_FOLDER']      = 'artifacts/uploads'
app.config['DEMO_FOLDER']        = 'artifacts/demo'
app.config['REPORTS_FOLDER']     = 'artifacts/reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS  = {'png', 'jpg', 'jpeg'}
HISTORY_FILE        = 'artifacts/history.json'
PATIENTS_FILE       = 'artifacts/patients.json'
MODEL_STATS_FILE    = 'artifacts/model_stats.json'

for folder in [
    app.config['UPLOAD_FOLDER'],
    app.config['DEMO_FOLDER'],
    app.config['REPORTS_FOLDER'],
    'artifacts'
]:
    os.makedirs(folder, exist_ok=True)

# ── Load Pipeline Once ────────────────────────────────
pipeline = PredictionPipeline()

# ── JSON Storage Helpers ──────────────────────────────

def _load_json(filepath: str, default) -> any:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default


def _save_json(filepath: str, data: any) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ── Image Helpers ─────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(img_array: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')


def draw_yolo_boxes(image: np.ndarray, yolo_result: dict) -> np.ndarray:
    img = image.copy()
    colors_map = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 0)}
    for i, box in enumerate(yolo_result.get("boxes", [])):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls   = yolo_result["class_ids"][i]
        name  = yolo_result["class_names"][i]
        score = yolo_result["scores"][i]
        color = colors_map.get(cls, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {score*100:.0f}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def draw_unet_mask(image: np.ndarray, unet_result: dict) -> np.ndarray:
    img  = image.copy()
    mask = unet_result.get("mask")
    if mask is None:
        return img
    mask_resized = cv2.resize(mask.astype(np.uint8),
                              (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask_resized > 0] = [0, 0, 200]
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    return img


# ── Model Stats Tracker ───────────────────────────────

def _get_model_stats() -> dict:
    default = {
        "cnn":  {"correct": 0, "total": 0, "accuracy": 0.0},
        "yolo": {"detections": 0, "avg_confidence": 0.0, "total": 0},
        "unet": {"segmented": 0, "avg_dice": 0.0, "avg_tumor_area": 0.0, "total": 0},
        "overall": {"total_scans": 0, "tumors_detected": 0, "accuracy": 0.0}
    }
    return _load_json(MODEL_STATS_FILE, default)


def _update_model_stats(report: dict) -> None:
    stats = _get_model_stats()

    # Ensure unet block always has all keys (backwards-compat with old JSON)
    stats.setdefault("unet", {})
    stats["unet"].setdefault("segmented", 0)
    stats["unet"].setdefault("avg_dice", 0.0)
    stats["unet"].setdefault("avg_tumor_area", 0.0)
    stats["unet"].setdefault("total", 0)

    stats["overall"]["total_scans"] += 1

    tumor_detected = report["classification"]["tumor_detected"]
    if tumor_detected:
        stats["overall"]["tumors_detected"] += 1

    # ── CNN accuracy: use confidence as proxy ──────────
    conf = report["classification"]["confidence"]
    stats["cnn"]["total"] += 1
    stats["cnn"]["correct"] += 1 if conf >= 0.50 else 0
    stats["cnn"]["accuracy"] = round(
        stats["cnn"]["correct"] / stats["cnn"]["total"] * 100, 1
    )

    # ── YOLO stats (only when tumor detected) ──────────
    loc = report.get("localization")
    if tumor_detected and loc and loc.get("num_regions", 0) > 0:
        stats["yolo"]["total"] += 1
        stats["yolo"]["detections"] += loc.get("num_regions", 0)
        raw_conf = (
            loc["regions"][0].get("confidence", 0.0)
            if loc.get("regions") else 0.0
        )
        if raw_conf > 1.0:
            raw_conf = raw_conf / 100.0
        new_conf_pct = round(raw_conf * 100, 2)
        old_avg      = stats["yolo"]["avg_confidence"]
        n            = stats["yolo"]["total"]
        stats["yolo"]["avg_confidence"] = round(
            (old_avg * (n - 1) + new_conf_pct) / n, 1
        )

    # ── U-Net stats (only when tumor detected) ─────────
    seg = report.get("segmentation")
    if tumor_detected and seg:
        n_prev = stats["unet"]["total"]
        stats["unet"]["total"]     += 1
        stats["unet"]["segmented"] += 1
        n_new = stats["unet"]["total"]

        raw_dice = seg.get("dice_score", 0.0)
        if raw_dice > 1.0:
            aw_dice = round(raw_dice / 100.0, 4)
        else:
            raw_dice = round(float(raw_dice), 4)

        stats["unet"]["avg_dice"] = round(
            (stats["unet"]["avg_dice"] * n_prev + raw_dice) / n_new, 4
        )

        area_pct = seg.get("tumor_area_pct", seg.get("coverage_percent", 0.0))
        if area_pct <= 1.0 and area_pct > 0:
            area_pct = round(area_pct * 100, 2)
        stats["unet"]["avg_tumor_area"] = round(
            (stats["unet"]["avg_tumor_area"] * n_prev + float(area_pct)) / n_new, 2
        )

        logger.info(
            f"[U-Net] dice={raw_dice:.4f}  area={area_pct:.2f}%  "
            f"segmented={stats['unet']['segmented']}  total={stats['unet']['total']}"
        )

    # ── Overall accuracy ────────────────────────────────
    t = stats["overall"]["total_scans"]
    stats["overall"]["accuracy"] = round(
        (stats["cnn"]["correct"] / t * 100) if t > 0 else 0.0, 1
    )

    _save_json(MODEL_STATS_FILE, stats)


# ── History Helpers ───────────────────────────────────

def _save_to_history(entry: dict) -> None:
    history = _load_json(HISTORY_FILE, [])
    history.insert(0, entry)
    history = history[:200]
    _save_json(HISTORY_FILE, history)


# ── Patient Helpers ───────────────────────────────────

def _get_patients() -> list:
    return _load_json(PATIENTS_FILE, [])


def _save_patient(patient: dict) -> None:
    patients = _get_patients()
    for i, p in enumerate(patients):
        if p["patient_id"] == patient["patient_id"]:
            patients[i] = patient
            _save_json(PATIENTS_FILE, patients)
            return
    patients.insert(0, patient)
    _save_json(PATIENTS_FILE, patients)


# ── Visualisations helper ─────────────────────────────

def _get_unet_mask_from_pipeline() -> np.ndarray | None:
    """
    FIX 1 — Robust U-Net cache access.

    Instead of blindly doing getattr(pipeline, '_unet_result_cache', None),
    we check multiple possible attribute names the pipeline might use and
    validate the result is actually a non-empty numpy array before using it.
    Returns the mask array, or None if not available.
    """
    # Try common attribute names the pipeline might store the mask under
    for attr in ('_unet_result_cache', '_unet_mask', 'unet_mask', '_last_unet_mask'):
        mask = getattr(pipeline, attr, None)
        if mask is not None:
            # Validate it's a real numpy array with content
            if isinstance(mask, np.ndarray) and mask.size > 0:
                return mask
    return None


def _build_visuals(image: np.ndarray, report: dict) -> tuple:
    """
    Correct inference flow:
      CNN → no tumor  → return only original MRI (yolo_b64=None, unet_b64=None)
      CNN → tumor     → run YOLO + U-Net → return all 3 images
    """
    original_b64 = encode_image_to_base64(image)
    yolo_b64     = None
    unet_b64     = None

    if report["classification"]["tumor_detected"]:
        # ── YOLO bounding-box overlay ──────────────────
        loc = report.get("localization")
        if loc and loc.get("num_regions", 0) > 0:
            yolo_boxes = {
                "boxes": [
                    [r["bounding_box"]["x1"], r["bounding_box"]["y1"],
                     r["bounding_box"]["x2"], r["bounding_box"]["y2"]]
                    for r in loc["regions"]
                ],
                "class_ids":   [0] * loc["num_regions"],
                "class_names": [r["class"] for r in loc["regions"]],
                "scores":      [r["confidence"] for r in loc["regions"]],
            }
            yolo_b64 = encode_image_to_base64(draw_yolo_boxes(image, yolo_boxes))

        # ── U-Net segmentation mask overlay ───────────
        # FIX 1: use robust helper instead of bare getattr
        if report.get("segmentation"):
            unet_mask = _get_unet_mask_from_pipeline()
            if unet_mask is not None:
                unet_b64 = encode_image_to_base64(
                    draw_unet_mask(image, {"mask": unet_mask})
                )
            else:
                logger.warning(
                    "[U-Net] Segmentation result present in report but no mask "
                    "array found on pipeline — skipping U-Net overlay."
                )

    return original_b64, yolo_b64, unet_b64


# ── PDF Report Generator ──────────────────────────────

def generate_pdf_report(scan_data: dict) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=22, textColor=colors.HexColor('#7C3AED'),
                                 spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle('Sub', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.grey,
                                 spaceAfter=16, alignment=TA_CENTER)

    story.append(Paragraph("🧠 NeuroScan AI — Diagnostic Report", title_style))
    story.append(Paragraph("AI-Powered Brain Tumor Detection System", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#7C3AED'), spaceAfter=16))

    ts = scan_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    info_data = [
        ["Scan ID",    scan_data.get("scan_id", "N/A"),
         "Date",       ts],
        ["Patient ID", scan_data.get("patient_id", "N/A"),
         "Patient",    scan_data.get("patient_name", "Anonymous")],
    ]
    info_table = Table(info_data, colWidths=[80, 170, 60, 170])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F0FF')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#F3F0FF')),
        ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
        ('PADDING',    (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 16))

    tumor_detected = scan_data.get("tumor_detected", False)
    result_color   = colors.HexColor('#DC2626') if tumor_detected else colors.HexColor('#16A34A')
    result_text    = "⚠ TUMOR DETECTED" if tumor_detected else "✓ NO TUMOR DETECTED"
    result_style   = ParagraphStyle('Result', parent=styles['Normal'],
                                    fontSize=16, textColor=result_color,
                                    fontName='Helvetica-Bold', alignment=TA_CENTER,
                                    borderPad=10)
    story.append(Paragraph(result_text, result_style))
    story.append(Spacer(1, 12))

    conf_data = [
        ["Diagnosis Label", scan_data.get("label", "N/A")],
        ["Confidence Score", f"{scan_data.get('confidence', 0):.1f}%"],
        ["Summary", scan_data.get("summary", "N/A")],
    ]
    conf_table = Table(conf_data, colWidths=[150, 330])
    conf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F3FF')),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#DDD6FE')),
        ('PADDING',    (0, 0), (-1, -1), 8),
        ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 16))

    loc = scan_data.get("localization")
    if loc and loc.get("num_regions", 0) > 0:
        story.append(Paragraph("Tumor Localization (YOLO Detection)",
                                ParagraphStyle('H2', parent=styles['Heading2'],
                                               textColor=colors.HexColor('#7C3AED'))))
        story.append(Spacer(1, 6))
        loc_rows = [["Region", "Class", "Confidence", "Bounding Box"]]
        for i, r in enumerate(loc.get("regions", []), 1):
            bb = r.get("bounding_box", {})
            loc_rows.append([
                str(i),
                r.get("class", "N/A"),
                f"{r.get('confidence', 0)*100:.1f}%",
                f"({bb.get('x1',0)}, {bb.get('y1',0)}) → ({bb.get('x2',0)}, {bb.get('y2',0)})"
            ])
        loc_table = Table(loc_rows, colWidths=[50, 120, 100, 210])
        loc_table.setStyle(TableStyle([
            ('BACKGROUND',     (0, 0), (-1, 0),  colors.HexColor('#7C3AED')),
            ('TEXTCOLOR',      (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTNAME',       (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE',       (0, 0), (-1, -1), 9),
            ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F7FF')]),
            ('PADDING',        (0, 0), (-1, -1), 6),
        ]))
        story.append(loc_table)
        story.append(Spacer(1, 16))

    seg = scan_data.get("segmentation")
    if seg:
        story.append(Paragraph("U-Net Segmentation Results",
                                ParagraphStyle('H2', parent=styles['Heading2'],
                                               textColor=colors.HexColor('#7C3AED'))))
        story.append(Spacer(1, 6))
        seg_data = [
            ["Tumor Area (px²)", str(seg.get("tumor_area_pixels", "N/A"))],
            ["Coverage (%)",     f"{seg.get('coverage_percent', seg.get('tumor_area_pct', 0)):.2f}%"],
            ["Dice Score",       f"{seg.get('dice_score', 0):.4f}"],
        ]
        seg_table = Table(seg_data, colWidths=[150, 330])
        seg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F3FF')),
            ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME',   (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE',   (0, 0), (-1, -1), 10),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#DDD6FE')),
            ('PADDING',    (0, 0), (-1, -1), 8),
        ]))
        story.append(seg_table)
        story.append(Spacer(1, 16))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.grey, spaceBefore=8, spaceAfter=8))
    disclaimer_style = ParagraphStyle('Disc', parent=styles['Normal'],
                                      fontSize=8, textColor=colors.grey,
                                      alignment=TA_CENTER)
    story.append(Paragraph(
        "This report is generated by NeuroScan AI for research/educational purposes only. "
        "It does not constitute medical advice. Always consult a qualified radiologist or neurosurgeon.",
        disclaimer_style
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NeuroScan AI v1.0",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ── Manual Predict ────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    FIX 2 — Proper JSON error responses instead of raising NeuroScanException
    which caused Flask to crash with a 500 and no JSON body.
    All exceptions are now caught and returned as { "error": "..." } JSON.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG"}), 400

        patient_id   = request.form.get('patient_id', '').strip() or f"PT-{uuid.uuid4().hex[:6].upper()}"
        patient_name = request.form.get('patient_name', 'Anonymous').strip()
        patient_age  = request.form.get('patient_age', 'N/A').strip()
        patient_sex  = request.form.get('patient_sex', 'N/A').strip()
        scan_type    = 'manual'

        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({"error": "Could not read image — file may be corrupt"}), 400

        mri_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Run prediction pipeline
        report = pipeline.run(mri_gray, image_path=filepath)

        # FIX 1: central visual builder uses robust mask accessor
        original_b64, yolo_b64, unet_b64 = _build_visuals(image, report)

        # Delete temp file AFTER pipeline + visuals
        if os.path.exists(filepath):
            os.remove(filepath)

        scan_entry = {
            "scan_id":        report["scan_id"],
            "timestamp":      report["timestamp"],
            "scan_type":      scan_type,
            "patient_id":     patient_id,
            "patient_name":   patient_name,
            "patient_age":    patient_age,
            "patient_sex":    patient_sex,
            "tumor_detected": report["classification"]["tumor_detected"],
            "label":          report["classification"]["label"],
            "confidence":     round(report["classification"]["confidence"] * 100, 2),
            "summary":        report["summary"],
            "localization":   report.get("localization") if report["classification"]["tumor_detected"] else None,
            "segmentation":   report.get("segmentation") if report["classification"]["tumor_detected"] else None,
        }

        # Persist
        _save_to_history({**scan_entry, "images": {"original": original_b64}})
        _save_patient({
            "patient_id":   patient_id,
            "patient_name": patient_name,
            "patient_age":  patient_age,
            "patient_sex":  patient_sex,
            "last_scan":    report["timestamp"],
            "scans":        1,
            "last_result":  report["classification"]["label"],
        })
        _update_model_stats(report)

        logger.info(
            f"[Manual] scan_id={scan_entry['scan_id']} "
            f"tumor={scan_entry['tumor_detected']} conf={scan_entry['confidence']}%"
        )

        return jsonify({
            **scan_entry,
            "images": {
                "original": original_b64,
                "yolo":     yolo_b64,
                "unet":     unet_b64,
            }
        }), 200

    except Exception as e:
        # FIX 2: catch ALL exceptions and return proper JSON — no more Flask crashes
        logger.error(f"[/predict] Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Demo Scan ─────────────────────────────────────────
@app.route('/demo/<sample_type>', methods=['GET'])
def demo_scan(sample_type):
    """
    FIX 2: All exceptions return proper JSON error responses.
    FIX 1: U-Net mask accessed via robust _get_unet_mask_from_pipeline().
    """
    try:
        demo_map = {
            "normal": os.path.join(app.config['DEMO_FOLDER'], 'normal.jpg'),
            "tumor":  os.path.join(app.config['DEMO_FOLDER'], 'tumor.jpg'),
        }
        if sample_type not in demo_map:
            return jsonify({"error": "Invalid demo type. Use 'normal' or 'tumor'"}), 400

        demo_path = demo_map[sample_type]
        if not os.path.exists(demo_path):
            demo_path_png = demo_path.replace('.jpg', '.png')
            if os.path.exists(demo_path_png):
                demo_path = demo_path_png
            else:
                return jsonify({"error": f"Demo image not found: {demo_path}"}), 404

        image = cv2.imread(demo_path)
        if image is None:
            return jsonify({"error": "Could not read demo image"}), 500

        mri_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Run prediction pipeline
        report = pipeline.run(mri_gray, image_path=demo_path)

        # FIX 1: central visual builder uses robust mask accessor
        original_b64, yolo_b64, unet_b64 = _build_visuals(image, report)

        scan_entry = {
            "scan_id":        report["scan_id"],
            "timestamp":      report["timestamp"],
            "scan_type":      "demo",
            "patient_id":     "DEMO",
            "patient_name":   "Demo Patient",
            "tumor_detected": report["classification"]["tumor_detected"],
            "label":          report["classification"]["label"],
            "confidence":     round(report["classification"]["confidence"] * 100, 2),
            "summary":        report["summary"],
            "localization":   report.get("localization") if report["classification"]["tumor_detected"] else None,
            "segmentation":   report.get("segmentation") if report["classification"]["tumor_detected"] else None,
        }

        # Persist history + model stats
        _save_to_history({**scan_entry, "images": {"original": original_b64}})
        _update_model_stats(report)

        logger.info(
            f"[Demo/{sample_type}] scan_id={scan_entry['scan_id']} "
            f"tumor={scan_entry['tumor_detected']} conf={scan_entry['confidence']}%"
        )

        return jsonify({
            **scan_entry,
            "images": {
                "original": original_b64,
                "yolo":     yolo_b64,
                "unet":     unet_b64,
            }
        }), 200

    except Exception as e:
        # FIX 2: proper JSON error — no more Flask crash
        logger.error(f"[/demo/{sample_type}] Demo scan failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Demo scan failed: {str(e)}"}), 500


# ── History ───────────────────────────────────────────
@app.route('/history', methods=['GET'])
def get_history():
    history = _load_json(HISTORY_FILE, [])
    slim = [{k: v for k, v in h.items() if k != 'images'} for h in history]
    return jsonify(slim), 200


@app.route('/history/<scan_id>', methods=['GET'])
def get_history_detail(scan_id):
    history = _load_json(HISTORY_FILE, [])
    for h in history:
        if h.get("scan_id") == scan_id:
            return jsonify(h), 200
    return jsonify({"error": "Scan not found"}), 404


@app.route('/history/<scan_id>', methods=['DELETE'])
def delete_history(scan_id):
    history = _load_json(HISTORY_FILE, [])
    history = [h for h in history if h.get("scan_id") != scan_id]
    _save_json(HISTORY_FILE, history)
    return jsonify({"message": "Deleted"}), 200


# ── Patients ──────────────────────────────────────────
@app.route('/patients', methods=['GET'])
def get_patients():
    return jsonify(_get_patients()), 200


@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.json or {}
    if not data.get("patient_id"):
        data["patient_id"] = f"PT-{uuid.uuid4().hex[:6].upper()}"
    data.setdefault("patient_name", "Unknown")
    data.setdefault("patient_age",  "N/A")
    data.setdefault("patient_sex",  "N/A")
    data.setdefault("last_scan",    None)
    data.setdefault("scans",        0)
    data.setdefault("last_result",  "N/A")
    _save_patient(data)
    return jsonify({"message": "Patient saved", "patient": data}), 201


@app.route('/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    patients = _get_patients()
    patients = [p for p in patients if p.get("patient_id") != patient_id]
    _save_json(PATIENTS_FILE, patients)
    return jsonify({"message": "Deleted"}), 200


# ── Model Performance ─────────────────────────────────
@app.route('/model-performance', methods=['GET'])
def model_performance():
    stats = _get_model_stats()

    stats.setdefault("unet", {})
    stats["unet"].setdefault("avg_dice",       0.0)
    stats["unet"].setdefault("avg_tumor_area", 0.0)
    stats["unet"].setdefault("segmented",      0)
    stats["unet"].setdefault("total",          0)

    return jsonify(stats), 200


# ── Report (PDF) ──────────────────────────────────────
@app.route('/report/<scan_id>', methods=['GET'])
def download_report(scan_id):
    history   = _load_json(HISTORY_FILE, [])
    scan_data = next((h for h in history if h.get("scan_id") == scan_id), None)

    if not scan_data:
        return jsonify({"error": "Scan not found"}), 404

    try:
        pdf_buffer = generate_pdf_report(scan_data)
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"NeuroScan_Report_{scan_id}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"[/report] Report generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Report generation failed"}), 500


# ── Reset Model Stats ────────────────────────────────
@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    clean = {
        "cnn":  {"correct": 0, "total": 0, "accuracy": 0.0},
        "yolo": {"detections": 0, "avg_confidence": 0.0, "total": 0},
        "unet": {"segmented": 0, "avg_dice": 0.0, "avg_tumor_area": 0.0, "total": 0},
        "overall": {"total_scans": 0, "tumors_detected": 0, "accuracy": 0.0}
    }
    _save_json(MODEL_STATS_FILE, clean)
    logger.info("model_stats.json reset to zero by /reset-stats")
    return jsonify({"message": "Stats reset successfully", "stats": clean}), 200


# ── Health ────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        "status":    "healthy",
        "system":    "NeuroScan-AI",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200


# ── Run ───────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("🧠 Starting NeuroScan-AI Flask App...")
    app.run(host='0.0.0.0', port=8000, debug=False)