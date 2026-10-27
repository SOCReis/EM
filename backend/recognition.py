"""
Reconhecimento facial com InsightFace (ArcFace).

Pipeline em 2 estágios para reconhecer rostos DISTANTES:

  1) DETECÇÃO
     - Imagens grandes usam TILING (fatias de det_size com overlap + NMS),
       para capturar rostos pequenos que sumiriam no resize global.
     - Imagens pequenas usam detecção única.

  2) EMBEDDING
     - Para cada rosto, faz crop com margem de 35%, aplica upscale LANCZOS4
       até o rosto ter FACE_UPSCALE_TO px, e RE-DETECTA no crop.
     - O embedding final vem deste crop limpo.

  3) MATCHING
     - Threshold ADAPTATIVO: rostos pequenos exigem score maior.
     - Compara contra TODAS as embeddings de TODOS os alunos e pega o
       máximo por aluno (multi-embedding).

Compatível com InsightFace 0.7.x e 1.x (prepare() é tolerante à diferença).
"""
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------- Configuração (env vars) ----------
DET_SIZE         = int(os.environ.get("DET_SIZE", "1024"))
DET_THRESH       = float(os.environ.get("DET_THRESH", "0.4"))
TILE_MIN_SIDE    = int(os.environ.get("TILE_MIN_SIDE", "1600"))
TILE_OVERLAP     = float(os.environ.get("TILE_OVERLAP", "0.2"))
MAX_TILES        = int(os.environ.get("MAX_TILES", "6"))

MIN_FACE_PX      = int(os.environ.get("MIN_FACE_PX", "36"))
FACE_UPSCALE_TO  = int(os.environ.get("FACE_UPSCALE_TO", "160"))
BLUR_MIN         = float(os.environ.get("BLUR_MIN", "15.0"))

REC_THRESHOLD      = float(os.environ.get("RECOGNITION_THRESHOLD", "0.42"))
REC_THRESHOLD_FAR  = float(os.environ.get("RECOGNITION_THRESHOLD_FAR", "0.55"))
FAR_FACE_PX        = int(os.environ.get("FAR_FACE_PX", "90"))

# ---------- Carrega o modelo (tolerante a 0.7.x e 1.x) ----------
_face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

try:
    _face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESH)
except TypeError:
    # algumas builds de 1.x removeram det_thresh do prepare
    _face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))

# garante o threshold no modelo de detecção (1.x lê daqui)
try:
    _face_app.det_model.det_thresh = DET_THRESH
except Exception:
    pass

print(f"[Recognition] InsightFace carregado · det_size={DET_SIZE} · "
      f"thresh={DET_THRESH} · min_face_px={MIN_FACE_PX}")


# ---------- Helpers de detecção ----------
def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(faces, iou_thresh=0.4):
    if len(faces) <= 1:
        return faces
    faces = sorted(faces,
                   key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                   reverse=True)
    keep = []
    for f in faces:
        if all(_iou(f.bbox, k.bbox) < iou_thresh for k in keep):
            keep.append(f)
    return keep


def _tile_has_content(tile):
    if tile.size == 0:
        return False
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    return float(gray.std()) >= 8.0


def _detect_tiled(img):
    H, W = img.shape[:2]
    det = DET_SIZE
    stride = max(1, int(det * (1 - TILE_OVERLAP)))

    xs = list(range(0, max(1, W - det + 1), stride))
    ys = list(range(0, max(1, H - det + 1), stride))
    if xs and xs[-1] + det < W: xs.append(W - det)
    if ys and ys[-1] + det < H: ys.append(H - det)
    xs = [max(0, x) for x in xs]
    ys = [max(0, y) for y in ys]
    coords = [(x, y) for y in ys for x in xs]
    if not coords:
        return _face_app.get(img)

    if len(coords) > MAX_TILES:
        idx = np.linspace(0, len(coords) - 1, MAX_TILES).astype(int)
        coords = [coords[i] for i in idx]

    all_faces = []
    for (x, y) in coords:
        tile = img[y:y+det, x:x+det]
        if tile.shape[0] < det or tile.shape[1] < det:
            continue
        if not _tile_has_content(tile):
            continue
        for f in _face_app.get(tile):
            f.bbox = f.bbox + np.array([x, y, x, y], dtype=np.float32)
            all_faces.append(f)

    return _nms(all_faces)


def _detect(img):
    H, W = img.shape[:2]
    if max(H, W) >= TILE_MIN_SIDE:
        return _detect_tiled(img)
    return _face_app.get(img)


def _reembed_from_crop(img, bbox):
    """Crop + upscale + re-detect → embedding limpo."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    if fw <= 0 or fh <= 0:
        return None

    margin = 0.35
    mw, mh = int(fw * margin), int(fh * margin)
    cx1, cy1 = max(0, x1 - mw), max(0, y1 - mh)
    cx2, cy2 = min(W, x2 + mw), min(H, y2 + mh)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    scale = max(1.0, FACE_UPSCALE_TO / max(fw, fh))
    if scale > 1.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)

    faces = _face_app.get(crop)
    if not faces:
        return None
    cf = max(faces,
             key=lambda c: (c.bbox[2]-c.bbox[0]) * (c.bbox[3]-c.bbox[1]))
    return cf.normed_embedding.astype(np.float32)


def _blur_score(img, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face = img[max(0, y1):y2, max(0, x1):x2]
    if face.size == 0:
        return 0.0
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------- API pública ----------
def analyze_image(bgr_img):
    """
    Pipeline completo. Retorna lista de dicts:
      { bbox:[x1,y1,x2,y2], embedding, face_px, blur, det_score, low_quality }
    """
    raw = _detect(bgr_img)
    results = []
    for f in raw:
        x1, y1, x2, y2 = f.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        face_px = min(fw, fh)
        if face_px < MIN_FACE_PX:
            continue

        blur = _blur_score(bgr_img, f.bbox)
        emb = _reembed_from_crop(bgr_img, f.bbox)
        if emb is None:
            emb = f.normed_embedding.astype(np.float32)

        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "embedding": emb,
            "face_px": int(face_px),
            "blur": round(blur, 1),
            "det_score": float(f.det_score),
            "low_quality": blur < BLUR_MIN,
        })
    return results


def threshold_for(face_px):
    """Threshold adaptativo: rosto pequeno → threshold maior."""
    if face_px >= FAR_FACE_PX:
        return REC_THRESHOLD
    t = (FAR_FACE_PX - face_px) / max(1, FAR_FACE_PX - MIN_FACE_PX)
    t = min(1.0, max(0.0, t))
    return REC_THRESHOLD + t * (REC_THRESHOLD_FAR - REC_THRESHOLD)


def match_against_db(embedding, db_rows, threshold):
    """
    db_rows: [{student_id, name, embedding}]
    Pega o máximo por aluno (multi-embedding).
    Retorna (match | None, best_score, second_score).
    """
    if not db_rows:
        return None, 0.0, 0.0

    per_student = {}
    for r in db_rows:
        sid = r["student_id"]
        score = float(np.dot(embedding, r["embedding"]))
        if sid not in per_student or score > per_student[sid][0]:
            per_student[sid] = (score, r["name"])

    ranked = sorted(per_student.items(), key=lambda kv: kv[1][0], reverse=True)
    if not ranked:
        return None, 0.0, 0.0

    (best_sid, (best_score, best_name)) = ranked[0]
    second = ranked[1][1][0] if len(ranked) > 1 else 0.0

    if best_score < threshold:
        return None, best_score, second
    return {"id": best_sid, "name": best_name}, best_score, second