import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from database import init_db
from recognition import (
    analyze_image,
    match_against_db,
    threshold_for,
    MIN_FACE_PX,
    DET_SIZE,
    REC_THRESHOLD,
    REC_THRESHOLD_FAR,
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)
CORS(app)

conn = init_db()

# ---------- Cache de embeddings ----------
_cache = {"data": None}
_cache_lock = threading.Lock()


def _invalidate_cache():
    with _cache_lock:
        _cache["data"] = None


def load_all_embeddings():
    with _cache_lock:
        if _cache["data"] is not None:
            return _cache["data"]
        rows = conn.execute("""
            SELECT e.student_id, s.name, e.embedding
            FROM face_embeddings e
            JOIN students s ON s.id = e.student_id
        """).fetchall()
        _cache["data"] = [
            {"student_id": r["student_id"], "name": r["name"],
             "embedding": np.frombuffer(r["embedding"], dtype=np.float32)}
            for r in rows
        ]
        return _cache["data"]


# ---------- Utilitários ----------
def bgr_from_upload(file_storage):
    data = file_storage.read()
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_max(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def upscale_if_small(img, min_side):
    h, w = img.shape[:2]
    if max(h, w) < min_side:
        s = min_side / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    return img


def _save_photo(img, name, suffix=""):
    fn = f"{int(datetime.now().timestamp()*1000)}_{name.replace(' ','_')}{suffix}.jpg"
    cv2.imwrite(str(UPLOAD_DIR / fn), img)
    return fn


def _best_from_image(img, for_registration=True):
    """Upscale para cadastro + analisa + retorna melhor face + foto salva."""
    if for_registration:
        img = upscale_if_small(img, 1000)
    else:
        img = resize_max(img, 2000)
    results = analyze_image(img)
    if not results:
        return None
    best = max(results,
               key=lambda r: (r["bbox"][2]-r["bbox"][0]) * (r["bbox"][3]-r["bbox"][1]))
    return best, img


# ---------- Rotas ----------
@app.route("/")
def index():
    return send_file(BASE_DIR / "index.html")


@app.route("/uploads/<path:filename>")
def serve_photo(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/students", methods=["GET"])
def list_students():
    rows = conn.execute("""
        SELECT s.id, s.name, s.created_at,
               COUNT(e.id) AS num_photos,
               (SELECT photo_path FROM face_embeddings
                WHERE student_id = s.id ORDER BY id LIMIT 1) AS photo_path
        FROM students s
        LEFT JOIN face_embeddings e ON e.student_id = s.id
        GROUP BY s.id
        ORDER BY s.name
    """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/students", methods=["POST"])
def add_student():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Nome é obrigatório"}), 400

    photos = request.files.getlist("photos") or request.files.getlist("photo")
    if not photos:
        return jsonify({"error": "Envie ao menos uma foto"}), 400

    embeddings = []
    for i, photo in enumerate(photos):
        img = bgr_from_upload(photo)
        if img is None:
            continue
        out = _best_from_image(img, for_registration=True)
        if not out:
            continue
        best, processed = out
        fn = _save_photo(processed, name, f"_{i}")
        embeddings.append({"embedding": best["embedding"], "photo_path": fn})

    if not embeddings:
        return jsonify({
            "error": f"Nenhum rosto utilizável detectado. O rosto precisa ter "
                     f"pelo menos {MIN_FACE_PX}px. Aproxime-se ou melhore a iluminação."
        }), 400

    cur = conn.execute("INSERT INTO students (name) VALUES (?)", (name,))
    sid = cur.lastrowid
    for e in embeddings:
        conn.execute(
            "INSERT INTO face_embeddings (student_id, embedding, photo_path) VALUES (?,?,?)",
            (sid, e["embedding"].tobytes(), e["photo_path"]),
        )
    conn.commit()
    _invalidate_cache()
    return jsonify({"id": sid, "name": name, "num_embeddings": len(embeddings)}), 201


@app.route("/api/students/<int:sid>/faces", methods=["POST"])
def add_face(sid):
    if not conn.execute("SELECT 1 FROM students WHERE id = ?", (sid,)).fetchone():
        return jsonify({"error": "Aluno não encontrado"}), 404

    photos = request.files.getlist("photos") or request.files.getlist("photo")
    if not photos:
        return jsonify({"error": "Envie ao menos uma foto"}), 400

    added = 0
    for i, photo in enumerate(photos):
        img = bgr_from_upload(photo)
        if img is None:
            continue
        out = _best_from_image(img, for_registration=True)
        if not out:
            continue
        best, processed = out
        fn = _save_photo(processed, f"aluno{sid}", f"_{i}")
        conn.execute(
            "INSERT INTO face_embeddings (student_id, embedding, photo_path) VALUES (?,?,?)",
            (sid, best["embedding"].tobytes(), fn),
        )
        added += 1
    conn.commit()
    _invalidate_cache()
    return jsonify({"ok": True, "added": added})


@app.route("/api/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn.execute("DELETE FROM students WHERE id = ?", (sid,))
    conn.commit()
    _invalidate_cache()
    return jsonify({"ok": True})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if "photo" not in request.files:
        return jsonify({"error": "Foto é obrigatória"}), 400

    img = bgr_from_upload(request.files["photo"])
    if img is None:
        return jsonify({"error": "Imagem inválida"}), 400

    img = upscale_if_small(img, 1280)
    img = resize_max(img, 2200)
    H, W = img.shape[:2]

    faces = analyze_image(img)
    if not faces:
        return jsonify({"detected": 0, "matches": []})

    db_rows = load_all_embeddings()
    matches = []
    for f in faces:
        thr = threshold_for(f["face_px"])
        match, score, second = match_against_db(f["embedding"], db_rows, thr)
        x1, y1, x2, y2 = f["bbox"]
        matches.append({
            "bbox_norm": [x1/W, y1/H, x2/W, y2/H],
            "face_px": f["face_px"],
            "blur": f["blur"],
            "low_quality": f["low_quality"],
            "match": match,
            "score": round(score, 3),
            "second_score": round(second, 3),
            "threshold": round(thr, 3),
        })

    return jsonify({"detected": len(faces), "matches": matches})


@app.route("/api/attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json() or {}
    sid = data.get("student_id")
    confidence = float(data.get("confidence", 1.0))
    if not sid:
        return jsonify({"error": "student_id é obrigatório"}), 400

    now = datetime.now()
    try:
        conn.execute(
            "INSERT INTO attendance (student_id, date, time, confidence) VALUES (?,?,?,?)",
            (sid, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), confidence),
        )
        conn.commit()
        return jsonify({"ok": True, "already_marked": False,
                        "time": now.strftime("%H:%M:%S")})
    except sqlite3.IntegrityError:
        return jsonify({"ok": True, "already_marked": True})


@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    rows = conn.execute("""
        SELECT a.id, a.student_id, s.name, a.date, a.time, a.confidence,
               (SELECT photo_path FROM face_embeddings
                WHERE student_id = s.id ORDER BY id LIMIT 1) AS photo_path
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (date,)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "det_size": DET_SIZE,
        "threshold_near": REC_THRESHOLD,
        "threshold_far": REC_THRESHOLD_FAR,
        "min_face_px": MIN_FACE_PX,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)