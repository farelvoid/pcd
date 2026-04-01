from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import uuid, cv2, os, mediapipe as mp, numpy as np

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

CHIN_TIP=152; FOREHEAD_TOP=10; LEFT_CHEEK=234; RIGHT_CHEEK=454
LEFT_JAW=172; RIGHT_JAW=397; LEFT_TEMPLE=127; RIGHT_TEMPLE=356

def classify_face_mediapipe(landmarks, img_w, img_h, hijab):
    def px(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])
    forehead=px(FOREHEAD_TOP); chin=px(CHIN_TIP)
    lc=px(LEFT_CHEEK); rc=px(RIGHT_CHEEK)
    lj=px(LEFT_JAW); rj=px(RIGHT_JAW)
    lt=px(LEFT_TEMPLE); rt=px(RIGHT_TEMPLE)
    face_height=np.linalg.norm(forehead-chin)
    cheek_width=np.linalg.norm(lc-rc)
    jaw_width=np.linalg.norm(lj-rj)
    temple_width=np.linalg.norm(lt-rt)
    ratio_h_w=face_height/cheek_width
    jaw_ratio=jaw_width/cheek_width
    temple_ratio=temple_width/cheek_width
    if hijab=="yes":
        if ratio_h_w>1.55: return "Oblong"
        elif ratio_h_w>1.30: return "Oval"
        elif jaw_ratio<0.75 and temple_ratio>0.90: return "Heart"
        else: return "Round"
    else:
        if ratio_h_w>1.60: return "Oblong"
        elif ratio_h_w>1.35 and jaw_ratio>0.80: return "Oval"
        elif jaw_ratio<0.75 and temple_ratio>0.90: return "Heart"
        elif jaw_ratio>0.90 and temple_ratio>0.90: return "Square"
        else: return "Round"

def draw_landmarks(img, landmarks, img_w, img_h):
    key_points=[CHIN_TIP,FOREHEAD_TOP,LEFT_CHEEK,RIGHT_CHEEK,LEFT_JAW,RIGHT_JAW,LEFT_TEMPLE,RIGHT_TEMPLE]
    for idx in key_points:
        lm=landmarks[idx]; cx,cy=int(lm.x*img_w),int(lm.y*img_h)
        cv2.circle(img,(cx,cy),5,(0,255,128),-1)
        cv2.circle(img,(cx,cy),7,(255,255,255),1)
    def pt(idx):
        lm=landmarks[idx]; return (int(lm.x*img_w),int(lm.y*img_h))
    cv2.line(img,pt(FOREHEAD_TOP),pt(CHIN_TIP),(255,200,0),2)
    cv2.line(img,pt(LEFT_CHEEK),pt(RIGHT_CHEEK),(0,180,255),2)
    cv2.line(img,pt(LEFT_JAW),pt(RIGHT_JAW),(200,0,255),2)
    cv2.line(img,pt(LEFT_TEMPLE),pt(RIGHT_TEMPLE),(0,255,200),2)
    return img

# Maksimal 3 rekomendasi per kombinasi
RECOMMENDATIONS = {
    "female": {
        "Oval": {
            "straight": ["Long layer","Bob sebahu","Curtain bangs"],
            "wavy":     ["Beach waves","Layer panjang","Soft wave bob"],
            "curly":    ["Curly layer","Shoulder cut","Natural curls"],
        },
        "Round": {
            "straight": ["Layer panjang","Side part","Long straight cut"],
            "wavy":     ["Wavy panjang","Volume atas","Loose waves"],
            "curly":    ["Curly panjang","Volume atas","Curly layer"],
        },
        "Square": {
            "straight": ["Soft layer","Poni tipis","Long straight"],
            "wavy":     ["Wavy medium","Textured waves","Beach waves"],
            "curly":    ["Loose curls","Layer ringan","Curly bob"],
        },
        "Oblong": {
            "straight": ["Poni depan","Medium cut","Blunt bob"],
            "wavy":     ["Wavy bob","Volume samping","Beach waves"],
            "curly":    ["Curly medium","Poni depan","Curly bob"],
        },
        "Heart": {
            "straight": ["Layer panjang","Poni samping","Long bob"],
            "wavy":     ["Wavy medium","Volume bawah","Beach waves"],
            "curly":    ["Curly layer","Loose curls","Natural curls"],
        },
    },
    "male": {
        "Oval": {
            "straight": ["Side part","Pompadour","Slick back"],
            "wavy":     ["Textured quiff","Messy waves","Side sweep"],
            "curly":    ["Curly top fade","Natural flow","Medium curls"],
        },
        "Round": {
            "straight": ["Undercut","High fade","Crew cut"],
            "wavy":     ["Messy textured","Wavy quiff","Textured quiff"],
            "curly":    ["Curly high fade","Volume atas","Curly top fade"],
        },
        "Square": {
            "straight": ["Textured crop","Crew cut","Side sweep"],
            "wavy":     ["Messy hair","Side part","Medium messy"],
            "curly":    ["Curly short cut","Taper fade","Curly top fade"],
        },
        "Oblong": {
            "straight": ["Fringe","Crew cut","Side sweep"],
            "wavy":     ["Medium messy","Side sweep","Wavy quiff"],
            "curly":    ["Curly medium","Curly fringe","Medium curls"],
        },
        "Heart": {
            "straight": ["Medium length","Natural flow","Side part"],
            "wavy":     ["Layered messy","Side part","Messy waves"],
            "curly":    ["Curly fringe","Medium curls","Natural flow"],
        },
    },
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file=request.files['image']
    gender=request.form.get('gender')
    hijab=request.form.get('hijab','no')
    hair_type=request.form.get('hair_type')
    if gender=="male": hijab="no"
    if not file: return redirect(url_for('index'))

    filename=secure_filename(file.filename)
    unique_name=str(uuid.uuid4())+"_"+filename
    filepath=os.path.join(app.config['UPLOAD_FOLDER'],unique_name)
    file.save(filepath)

    img=cv2.imread(filepath)
    if img is None:
        return render_template('result.html', error="Gambar tidak bisa dibaca.",
            image_path=filepath, annotated_path=None, face_shape="Error",
            recommendations=[], gender=gender, hijab=hijab, hair_type=hair_type)

    img_h,img_w=img.shape[:2]
    results=face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return render_template('result.html', error="Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas.",
            image_path=filepath, annotated_path=None, face_shape="Tidak Terdeteksi",
            recommendations=[], gender=gender, hijab=hijab, hair_type=hair_type)

    landmarks=results.multi_face_landmarks[0].landmark
    face_shape=classify_face_mediapipe(landmarks,img_w,img_h,hijab)

    gender_key="female" if gender=="female" else "male"
    recommendations=RECOMMENDATIONS.get(gender_key,{}).get(face_shape,{}).get(hair_type,["Konsultasikan dengan stylist"])

    img_annotated=draw_landmarks(img.copy(),landmarks,img_w,img_h)
    annotated_name="annotated_"+unique_name
    annotated_path=os.path.join(app.config['UPLOAD_FOLDER'],annotated_name)
    cv2.imwrite(annotated_path,img_annotated)

    return render_template('result.html', error=None,
        image_path=filepath, annotated_path=annotated_path,
        face_shape=face_shape, recommendations=recommendations,
        gender=gender, hijab=hijab, hair_type=hair_type)

if __name__=='__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
