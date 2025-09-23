from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, send_file, flash
import cv2, os, threading, time, sqlite3, pickle
from queue import Queue
from datetime import datetime
from deepface import DeepFace
import numpy as np
import base64, re
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import shutil

app = Flask(__name__)
app.secret_key = "supersecret123"  # ganti dengan string random lebih aman

# ------------------ FOLDER & DB ------------------
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

DB_PATH = "db/attendance.db"
os.makedirs("db", exist_ok=True)

EMB_FILE = "face_embeddings.pkl"

# DB init
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                course TEXT,
                datetime TEXT
            )""")
conn.commit()
conn.close()

# ------------------ WEBCAM ------------------
cap = None
camera_active = False
frame_queue = Queue(maxsize=1)
last_face_names = []  # simpan wajah terakhir untuk tombol presensi

# ------------------ EMBEDDINGS ------------------
def save_embeddings():
    """Simpan embeddings ke file pickle"""
    with open(EMB_FILE, "wb") as f:
        pickle.dump(face_embeddings, f)

def load_face_embeddings():
    """Load embeddings dari pickle kalau ada, kalau tidak generate dari folder"""
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            print("[INFO] Embeddings loaded dari pickle")
            return pickle.load(f)
    else:
        embeddings_db = {}
        for person in os.listdir(FACE_DIR):
            person_path = os.path.join(FACE_DIR, person)
            if os.path.isdir(person_path):
                embeddings_db[person] = []
                for f in os.listdir(person_path):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        path = os.path.join(person_path, f)
                        try:
                            emb = DeepFace.represent(
                                img_path=path, model_name='Facenet', enforce_detection=False
                            )[0]['embedding']
                            embeddings_db[person].append(np.array(emb))
                        except Exception as e:
                            print("Error embedding:", f, e)
        with open(EMB_FILE, "wb") as f:
            pickle.dump(embeddings_db, f)
        print("[INFO] Embeddings baru digenerate & disimpan ke pickle")
        return embeddings_db

face_embeddings = load_face_embeddings()

# ------------------ CAMERA THREAD ------------------
def camera_loop():
    global cap, camera_active, frame_queue
    while True:
        if camera_active:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
            success, frame = cap.read()
            if success:
                if not frame_queue.empty():
                    try: frame_queue.get_nowait()
                    except: pass
                frame_queue.put(frame)
        else:
            time.sleep(0.05)

threading.Thread(target=camera_loop, daemon=True).start()

# ------------------ FRAME GENERATOR ------------------
def gen_frames():
    global frame_queue, last_face_names
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while camera_active:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

                current_face_names = []

                for (x, y, w, h) in faces:
                    try:
                        face_img = frame[y:y+h, x:x+w]
                        emb = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']

                        best_match = "Unknown"
                        best_score = -1
                        # bandingkan dengan semua embedding
                        for name, emb_list in face_embeddings.items():
                            for ref_emb in emb_list:
                                score = cosine_similarity([emb], [ref_emb])[0][0]
                                if score > best_score:
                                    best_score = score
                                    best_match = name
                        if best_score < 0.75:
                            best_match = "Unknown"

                        current_face_names.append((x, y, w, h, best_match))
                    except Exception as e:
                        print("DeepFace error:", e)
                        current_face_names.append((x, y, w, h, "Unknown"))

                last_face_names = current_face_names  # update wajah terakhir

                # Draw box & label
                for (x, y, w, h, name) in current_face_names:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print("Frame error:", e)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.01)

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global camera_active, cap, frame_queue
    camera_active = False
    if cap and cap.isOpened():
        cap.release()
        cap = None
    while not frame_queue.empty():
        frame_queue.get()
    return "Camera stopped"

# ------------------ ATTENDANCE ------------------
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ambil daftar mata kuliah & tanggal unik
    c.execute("SELECT DISTINCT course FROM attendance")
    courses = [row[0] for row in c.fetchall()]
    c.execute("SELECT DISTINCT DATE(datetime) FROM attendance")
    dates = [row[0] for row in c.fetchall()]

    selected_course = request.form.get("course")
    selected_date = request.form.get("date")

    if selected_course and selected_date:
        c.execute("""SELECT name, course, datetime 
                     FROM attendance 
                     WHERE course=? AND DATE(datetime)=? 
                     ORDER BY datetime DESC""", 
                  (selected_course, selected_date))
    else:
        c.execute("""SELECT name, course, datetime 
                     FROM attendance 
                     ORDER BY datetime DESC""")

    records = c.fetchall()
    conn.close()

    return render_template("attendance.html",
                           courses=courses, dates=dates,
                           selected_course=selected_course,
                           selected_date=selected_date,
                           records=records)

@app.route('/download/<course>/<date>')
def download_csv_date(course, date):
    import csv
    path = f"db/{course}_{date}_attendance.csv"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT name, course, datetime 
                 FROM attendance 
                 WHERE course=? AND DATE(datetime)=? 
                 ORDER BY datetime DESC""", (course, date))
    rows = c.fetchall()
    conn.close()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Nama", "Mata Kuliah", "Tanggal & Waktu"])
        writer.writerows(rows)

    return send_file(path, as_attachment=True)

# ------------------ PRESENSI BUTTON ------------------
@app.route('/rekam_presensi', methods=['POST'])
def rekam_presensi():
    global last_face_names
    data = request.get_json()
    course = data.get("course")

    if not course:
        return jsonify({"success": False, "messages": ["Mata kuliah belum dipilih"]})

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pesan = []

    for (_, _, _, _, name) in last_face_names:
        if name != "Unknown":
            today = datetime.now().strftime("%Y-%m-%d")
            c.execute("""SELECT * FROM attendance 
                         WHERE name=? AND course=? AND DATE(datetime)=?""", 
                      (name, course, today))
            already = c.fetchone()
            if not already:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO attendance (name, course, datetime) VALUES (?, ?, ?)", 
                          (name, course, now))
                pesan.append(f"{name} berhasil absen {course} pada {now}")

    conn.commit()
    conn.close()

    if pesan:
        return jsonify({"success": True, "messages": pesan})
    else:
        return jsonify({"success": False, "messages": ["Tidak ada wajah baru terdeteksi / sudah absen hari ini"]})

# ------------------ MANAGE ------------------
@app.route('/manage')
def manage():
    persons = []
    if os.path.exists(FACE_DIR):
        persons = [d for d in os.listdir(FACE_DIR) if os.path.isdir(os.path.join(FACE_DIR, d))]
    return render_template("manage.html", persons=persons)

@app.route('/delete_person', methods=['POST'])
def delete_person():
    name = request.form['name']
    folder_path = os.path.join(FACE_DIR, name)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        flash(f"Folder wajah {name} berhasil dihapus.", "success")
        if name in face_embeddings:
            del face_embeddings[name]  # hapus embedding juga
            save_embeddings()
    else:
        flash(f"Folder wajah {name} tidak ditemukan.", "error")

    return redirect(url_for('manage'))

@app.route('/add_face_cam', methods=['POST'])
def add_face_cam():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')
    if not name or not image_data:
        return "Data tidak lengkap", 400
    try:
        save_dir = os.path.join(FACE_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        existing = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        filename = f"{name}_{existing+1}.jpg"

        img_str = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        cv_img = np.array(img)
        cv_img = cv_img[:, :, ::-1].copy()  # RGB -> BGR
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = cv_img[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (160, 160))
            cv2.imwrite(os.path.join(save_dir, filename), face_crop)
        else:
            img.save(os.path.join(save_dir, filename))

        # update embeddings on-the-fly
        emb = DeepFace.represent(
            img_path=os.path.join(save_dir, filename), 
            model_name='Facenet', enforce_detection=False
        )[0]['embedding']

        if name not in face_embeddings:
            face_embeddings[name] = []
        face_embeddings[name].append(np.array(emb))

        save_embeddings()

        return jsonify({"success": True, "count": existing+1}), 200
    except Exception as e:
        print("Error add_face_cam:", e)
        return "Gagal menyimpan wajah", 500

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
