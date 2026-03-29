from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import cv2
import os
import mediapipe as mp
import numpy as np

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── MediaPipe Face Mesh ───────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ─── Indeks landmark yang dipakai ─────────────────────────────────────────────
CHIN_TIP       = 152
FOREHEAD_TOP   = 10
LEFT_CHEEK     = 234
RIGHT_CHEEK    = 454
LEFT_JAW       = 172
RIGHT_JAW      = 397
LEFT_TEMPLE    = 127
RIGHT_TEMPLE   = 356


def classify_face_mediapipe(landmarks, img_w, img_h, hijab):
    def px(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    forehead    = px(FOREHEAD_TOP)
    chin        = px(CHIN_TIP)
    left_cheek  = px(LEFT_CHEEK)
    right_cheek = px(RIGHT_CHEEK)
    left_jaw    = px(LEFT_JAW)
    right_jaw   = px(RIGHT_JAW)
    left_temple = px(LEFT_TEMPLE)
    right_temple= px(RIGHT_TEMPLE)

    face_height   = np.linalg.norm(forehead - chin)
    cheek_width   = np.linalg.norm(left_cheek - right_cheek)
    jaw_width     = np.linalg.norm(left_jaw - right_jaw)
    temple_width  = np.linalg.norm(left_temple - right_temple)

    ratio_h_w    = face_height / cheek_width
    jaw_ratio    = jaw_width / cheek_width
    temple_ratio = temple_width / cheek_width

    if hijab == "yes":
        if ratio_h_w > 1.55:
            return "Oblong"
        elif ratio_h_w > 1.30:
            return "Oval"
        elif jaw_ratio < 0.75 and temple_ratio > 0.90:
            return "Heart"
        else:
            return "Round"
    else:
        if ratio_h_w > 1.60:
            return "Oblong"
        elif ratio_h_w > 1.35 and jaw_ratio > 0.80:
            return "Oval"
        elif jaw_ratio < 0.75 and temple_ratio > 0.90:
            return "Heart"
        elif jaw_ratio > 0.90 and temple_ratio > 0.90:
            return "Square"
        else:
            return "Round"


def draw_landmarks(img, landmarks, img_w, img_h):
    key_points = [CHIN_TIP, FOREHEAD_TOP, LEFT_CHEEK, RIGHT_CHEEK,
                  LEFT_JAW, RIGHT_JAW, LEFT_TEMPLE, RIGHT_TEMPLE]
    for idx in key_points:
        lm = landmarks[idx]
        cx, cy = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 128), -1)
        cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)

    def pt(idx):
        lm = landmarks[idx]
        return (int(lm.x * img_w), int(lm.y * img_h))

    cv2.line(img, pt(FOREHEAD_TOP), pt(CHIN_TIP),    (255, 200, 0),  2)
    cv2.line(img, pt(LEFT_CHEEK),   pt(RIGHT_CHEEK),  (0, 180, 255), 2)
    cv2.line(img, pt(LEFT_JAW),     pt(RIGHT_JAW),    (200, 0, 255), 2)
    cv2.line(img, pt(LEFT_TEMPLE),  pt(RIGHT_TEMPLE), (0, 255, 200), 2)
    return img


def get_recommendation(face_shape, gender, hair_type):
    if gender == "female":
        data = {
            "Oval":   {"straight": "Long layer, bob sebahu, curtain bangs, sleek straight",
                       "wavy":     "Beach waves, layer panjang, soft wave bob",
                       "curly":    "Curly layer, shoulder cut, natural curls"},
            "Round":  {"straight": "Layer panjang, side part, long straight cut",
                       "wavy":     "Wavy panjang, volume atas, loose waves",
                       "curly":    "Curly panjang, hindari pendek, volume atas"},
            "Square": {"straight": "Soft layer, poni tipis, long straight",
                       "wavy":     "Wavy medium, textured waves",
                       "curly":    "Loose curls, layer ringan, curly bob"},
            "Oblong": {"straight": "Poni depan, medium cut, blunt bob",
                       "wavy":     "Wavy bob, volume samping",
                       "curly":    "Curly medium, poni depan, curly bob"},
            "Heart":  {"straight": "Layer panjang, poni samping, long bob",
                       "wavy":     "Wavy medium, volume bawah",
                       "curly":    "Curly layer, hindari volume atas"},
        }
    else:
        data = {
            "Oval":   {"straight": "Side part, pompadour, slick back",
                       "wavy":     "Textured quiff, messy waves",
                       "curly":    "Curly top fade, natural curls"},
            "Round":  {"straight": "Undercut + volume atas, high fade",
                       "wavy":     "Messy textured, wavy quiff",
                       "curly":    "Curly high fade, volume atas"},
            "Square": {"straight": "Textured crop, crew cut",
                       "wavy":     "Messy hair, side part",
                       "curly":    "Curly short cut, taper fade"},
            "Oblong": {"straight": "Fringe, crew cut, low volume",
                       "wavy":     "Medium messy, side sweep",
                       "curly":    "Curly medium, fringe"},
            "Heart":  {"straight": "Medium length, natural flow",
                       "wavy":     "Layered messy, side part",
                       "curly":    "Curly fringe, medium curls"},
        }
    return data.get(face_shape, {}).get(hair_type, "Rekomendasi tidak tersedia")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    file      = request.files['image']
    gender    = request.form.get('gender')
    hijab     = request.form.get('hijab', 'no')
    hair_type = request.form.get('hair_type')

    if gender == "male":
        hijab = "no"

    if not file:
        return redirect(url_for('index'))

    filename    = secure_filename(file.filename)
    unique_name = str(uuid.uuid4()) + "_" + filename
    filepath    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return render_template('result.html',
                               image_path=filepath,
                               annotated_path=None,
                               face_shape="Error",
                               recommendation="Gambar tidak bisa dibaca",
                               hair_type=hair_type)

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return render_template('result.html',
                               image_path=filepath,
                               annotated_path=None,
                               face_shape="Tidak terdeteksi",
                               recommendation="Pastikan wajah terlihat jelas dan pencahayaan cukup",
                               hair_type=hair_type)

    landmarks = results.multi_face_landmarks[0].landmark

    face_shape     = classify_face_mediapipe(landmarks, img_w, img_h, hijab)
    recommendation = get_recommendation(face_shape, gender, hair_type)

    img_annotated  = draw_landmarks(img.copy(), landmarks, img_w, img_h)
    annotated_name = "annotated_" + unique_name
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
    cv2.imwrite(annotated_path, img_annotated)

    return render_template('result.html',
                           image_path=filepath,
                           annotated_path=annotated_path,
                           face_shape=face_shape,
                           recommendation=recommendation,
                           hair_type=hair_type)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
