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
import pydicom
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

ALLOWED_EXTENSIONS  = {'png', 'jpg', 'jpeg', 'dcm'}
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


def load_dicom_image(filepath: str) -> np.ndarray:
    """Convert DICOM file to BGR numpy array for OpenCV compatibility."""
    ds = pydicom.dcmread(filepath, force=True)

    # Fix missing file meta fields
    if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Fix missing required pixel fields
    if not hasattr(ds, 'PhotometricInterpretation'):
        ds.PhotometricInterpretation = "MONOCHROME2"
    if not hasattr(ds, 'PixelRepresentation'):
        ds.PixelRepresentation = 0

    pixel_array = ds.pixel_array.astype(np.float32)
    # Normalize pixel values to 0-255 range
    pixel_array = ((pixel_array - pixel_array.min()) /
                   (pixel_array.max() - pixel_array.min()) * 255)
    img = pixel_array.astype(np.uint8)
    # Convert grayscale to BGR so rest of pipeline works normally
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
    old_avg = stats["cnn"].get("accuracy", 0.0)
    n = stats["cnn"]["total"]
    stats["cnn"]["accuracy"] = round(
        (old_avg * (n - 1) + conf * 100) / n, 1
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
            raw_dice = round(raw_dice / 100.0, 4)
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
    # t = stats["overall"]["total_scans"]
    # ── Overall accuracy ────────────────────────────────
    cnn_acc = stats["cnn"].get("accuracy", 0.0)
    yolo_acc = stats["yolo"].get("avg_confidence", 0.0)
    unet_acc = stats["unet"].get("avg_dice", 0.0) * 100  # dice score -> %

    stats["overall"]["accuracy"] = round(
        (cnn_acc + yolo_acc + unet_acc) / 3, 1)

    _save_json(MODEL_STATS_FILE, stats)


# ── History Helpers  ───────────────────────────────────

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
    Robust U-Net cache access.
    Checks multiple possible attribute names the pipeline might use and
    validates the result is actually a non-empty numpy array before using it.
    Returns the mask array, or None if not available.
    """
    for attr in ('_unet_result_cache', '_unet_mask', 'unet_mask', '_last_unet_mask'):
        mask = getattr(pipeline, attr, None)
        if mask is not None:
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
    story.append(Paragraph("🤖 AI-Powered Brain Tumor Detection System", sub_style))
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHANGE 1: MRI Scan Images section (Original, YOLO, U-Net)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    images = scan_data.get("images", {})
    original_b64 = images.get("original")
    yolo_b64     = images.get("yolo")
    unet_b64     = images.get("unet")

    if original_b64 or yolo_b64 or unet_b64:
        story.append(Paragraph("MRI Scan Images",
                                ParagraphStyle('H2', parent=styles['Heading2'],
                                               textColor=colors.HexColor('#7C3AED'))))
        story.append(Spacer(1, 8))
        img_cells, img_labels = [], []
        for b64, label in [
            (original_b64, "Original MRI"),
            (yolo_b64,     "YOLO Detection"),
            (unet_b64,     "U-Net Segmentation")
        ]:
            if b64:
                img_io = BytesIO(base64.b64decode(b64))
                img_cells.append(RLImage(img_io, width=1.8*inch, height=1.8*inch))
                img_labels.append(Paragraph(label, ParagraphStyle('IL',
                    parent=styles['Normal'], fontSize=8,
                    textColor=colors.HexColor('#7C3AED'), alignment=TA_CENTER)))
        while len(img_cells) < 3:
            img_cells.append('')
            img_labels.append('')
        img_table = Table([img_cells, img_labels],
                          colWidths=[2.0*inch, 2.0*inch, 2.0*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING',    (0, 0), (-1, -1), 6),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
            ('BACKGROUND', (0, 0), (-1,  0), colors.HexColor('#F5F3FF')),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 16))
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5,
                        color=colors.HexColor('#7C3AED'), spaceAfter=6))

    author_style = ParagraphStyle('Author', parent=styles['Normal'],
                               fontSize=9, textColor=colors.HexColor('#7C3AED'),
                               alignment=TA_CENTER, fontName='Helvetica-Bold')

    story.append(Paragraph("👨🏻‍🎓 Developed by Kaushik Chariya 👨🏻‍🎓", author_style))

    contact_style = ParagraphStyle('Contact', parent=styles['Normal'],
                                fontSize=8, textColor=colors.grey,
                                alignment=TA_CENTER)

    story.append(Paragraph(
        'chariyajkaushik1435@gmail.com  |  '
        '<link href="https://kaushik-chariya.netlify.app">kaushik-chariya.netlify.app</link>',
        contact_style
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
    Accepts PNG, JPG, JPEG, and DCM (DICOM) files for tumor detection.
    All exceptions are caught and returned as { "error": "..." } JSON.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, DCM"}), 400

        patient_id   = request.form.get('patient_id', '').strip() or f"PT-{uuid.uuid4().hex[:6].upper()}"
        patient_name = request.form.get('patient_name', 'Anonymous').strip()
        patient_age  = request.form.get('patient_age', 'N/A').strip()
        patient_sex  = request.form.get('patient_sex', 'N/A').strip()
        scan_type    = 'manual'

        # Preserve original extension for DICOM detection
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{original_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load image — DICOM handled separately
        if original_ext == 'dcm':
            image = load_dicom_image(filepath)
            # YOLO DCM accept nahi karta — PNG mein convert karke save karo
            png_filepath = filepath.replace('.dcm', '.png')
            cv2.imwrite(png_filepath, image)
            yolo_path = png_filepath
        else:
            image = cv2.imread(filepath)
            yolo_path = filepath

        if image is None:
            os.remove(filepath)
            return jsonify({"error": "Could not read image — file may be corrupt"}), 400

        mri_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Run prediction pipeline
        report = pipeline.run(mri_gray, image_path=yolo_path)

        # Build visuals using robust mask accessor
        original_b64, yolo_b64, unet_b64 = _build_visuals(image, report)

        # Delete temp file AFTER pipeline + visuals
        if os.path.exists(filepath):
            os.remove(filepath)
        if original_ext == 'dcm' and os.path.exists(png_filepath):
            os.remove(png_filepath)

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

        # Persist (history mein sirf original save hoti hai — yolo/unet RAM-only)
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
        logger.error(f"[/predict] Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Demo Scan ─────────────────────────────────────────
@app.route('/demo/<sample_type>', methods=['GET'])
def demo_scan(sample_type):
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

        # Build visuals using robust mask accessor
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHANGE 2: /report route — GET + POST dono support karta hai
# POST body mein images inject karo (yolo + unet) PDF ke liye
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route('/report/<scan_id>', methods=['GET', 'POST'])
def download_report(scan_id):
    history   = _load_json(HISTORY_FILE, [])
    scan_data = next((h for h in history if h.get("scan_id") == scan_id), None)

    if not scan_data:
        return jsonify({"error": "Scan not found"}), 404

    # POST body mein images inject karo agar available hain
    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
        if body.get("images"):
            scan_data = {**scan_data, "images": body["images"]}

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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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