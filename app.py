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
app.secret_key = "supersecret123"  # secret key untuk session & flash message
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ------------------ FOLDER & DB ------------------
FACE_DIR = "faces"  # folder untuk simpan gambar wajah
os.makedirs(FACE_DIR, exist_ok=True)

DB_PATH = "db/attendance.db"  # lokasi database SQLite
os.makedirs("db", exist_ok=True)

EMB_FILE = "face_embeddings.pkl"  # file pickle untuk simpan embedding wajah
# ------------------ ATTENDANCE PHOTO ------------------
ATT_PHOTO_DIR = "attendance_photos"
os.makedirs(ATT_PHOTO_DIR, exist_ok=True)

# Inisialisasi database SQLite
conn = get_db()
c = conn.cursor()

# --- create table jika belum ada ---
c.execute("""CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    course TEXT,
    datetime TEXT
)""")

# --- cek apakah kolom 'class' sudah ada ---
c.execute("PRAGMA table_info(attendance)")
cols = [row[1] for row in c.fetchall()]

if "class" not in cols:
    print("🔧 Menambahkan kolom 'class' ke tabel attendance...")
    c.execute("ALTER TABLE attendance ADD COLUMN class TEXT")

conn.commit()
conn.close()


# ------------------ WEBCAM ------------------
cap = None  # objek kamera OpenCV
camera_active = False  # status kamera
frame_queue = Queue(maxsize=1)  # buffer frame agar tidak penuh
last_face_names = []  # simpan wajah terakhir terdeteksi (untuk presensi manual)

# ------------------ EMBEDDINGS ------------------
def save_embeddings():
    """Simpan embeddings wajah ke file pickle"""
    with open(EMB_FILE, "wb") as f:
        pickle.dump(face_embeddings, f)

def load_face_embeddings():
    """Load embeddings dari pickle. Kalau tidak ada, generate dari folder 'faces'"""
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            print("[INFO] Embeddings loaded dari pickle")
            return pickle.load(f)
    else:
        embeddings_db = {}
        # Loop setiap folder nama orang
        for person in os.listdir(FACE_DIR):
            person_path = os.path.join(FACE_DIR, person)
            if os.path.isdir(person_path):
                embeddings_db[person] = []
                # Loop file gambar dalam folder orang tsb
                for f in os.listdir(person_path):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        path = os.path.join(person_path, f)
                        try:
                            # generate embedding dengan DeepFace
                            emb = DeepFace.represent(
                                img_path=path, model_name='Facenet', enforce_detection=False
                            )[0]['embedding']
                            embeddings_db[person].append(np.array(emb))
                        except Exception as e:
                            print("Error embedding:", f, e)
        # simpan embeddings baru ke pickle
        with open(EMB_FILE, "wb") as f:
            pickle.dump(embeddings_db, f)
        print("[INFO] Embeddings baru digenerate & disimpan ke pickle")
        return embeddings_db

face_embeddings = load_face_embeddings()

# ------------------ CAMERA THREAD ------------------
def camera_loop():
    """Thread untuk baca frame dari kamera secara terus-menerus"""
    global cap, camera_active, frame_queue
    while True:
        if camera_active:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FPS, 60)
            success, frame = cap.read()
            if success:
                # pastikan queue tidak penuh
                if not frame_queue.empty():
                    try: frame_queue.get_nowait()
                    except: pass
                frame_queue.put(frame)
        else:
            time.sleep(0.06)  # hemat CPU saat kamera off

threading.Thread(target=camera_loop, daemon=True).start()

# ------------------ FRAME GENERATOR ------------------
def gen_frames():
    """Generator frame dengan deteksi wajah real-time"""
    global frame_queue, last_face_names

    while camera_active:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                # Deteksi wajah pakai DeepFace
                detections = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="ssd",  # pilihan detector (ssd, yolov8, yolov11)
                    enforce_detection=False
                )

                current_face_names = []

                for det in detections:
                    # Ambil bounding box
                    facial_area = det["facial_area"]
                    x, y, w, h = (
                        facial_area["x"],
                        facial_area["y"],
                        facial_area["w"],
                        facial_area["h"]
                    )
                    face_img = det["face"]

                    # Buat embedding dari wajah terdeteksi
                    emb = DeepFace.represent(
                        img_path=face_img,
                        model_name="Facenet",
                        detector_backend="skip",  # sudah crop, skip deteksi ulang
                        enforce_detection=False
                    )[0]["embedding"]

                    # Cocokkan dengan database embeddings
                    best_match = "Unknown"
                    best_score = -1
                    for name, emb_list in face_embeddings.items():
                        for ref_emb in emb_list:
                            score = cosine_similarity([emb], [ref_emb])[0][0]
                            if score > best_score:
                                best_score = score
                                best_match = name
                    if best_score < 0.75:  # threshold kemiripan
                        best_match = "Unknown"

                    current_face_names.append((x, y, w, h, best_match))

                last_face_names = current_face_names

                # Gambar kotak & nama di frame
                for (x, y, w, h, name) in current_face_names:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print("Frame error:", e)

            # Encode frame ke JPEG & kirim ke browser
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        else:
            time.sleep(0.01)


# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template('index.html')  # halaman utama

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True  # nyalakan kamera
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    """Hentikan kamera & kosongkan queue"""
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
    conn = get_db()
    c = conn.cursor()

    # ambil daftar unik
    c.execute("SELECT DISTINCT course FROM attendance WHERE course IS NOT NULL")
    courses = [r[0] for r in c.fetchall()]

    c.execute("SELECT DISTINCT DATE(datetime) FROM attendance")
    dates = [r[0] for r in c.fetchall()]

    c.execute("SELECT DISTINCT class FROM attendance WHERE class IS NOT NULL")
    classes = [r[0] for r in c.fetchall()]

    # ambil filter dari form
    selected_course = request.form.get("course")
    selected_date = request.form.get("date")
    selected_class = request.form.get("class")

    # bangun query dinamis
    query = """
        SELECT name, course, class, datetime
        FROM attendance
        WHERE 1=1
    """
    params = []

    if selected_course:
        query += " AND course=?"
        params.append(selected_course)

    if selected_date:
        query += " AND datetime LIKE ?"
        params.append(f"{selected_date}%")

    if selected_class:
        query += " AND class=?"
        params.append(selected_class)

    query += " ORDER BY datetime DESC"

    c.execute(query, params)
    records = c.fetchall()

    conn.close()

    return render_template(
        "attendance.html",
        courses=courses,
        dates=dates,
        classes=classes,
        selected_course=selected_course,
        selected_date=selected_date,
        selected_class=selected_class,
        records=records
    )

@app.route("/delete_attendance", methods=["POST"])
def delete_attendance():
    data = request.get_json()
    name = data.get("name")
    course = data.get("course")
    date = data.get("date")
    class_name = data.get("class")

    if not name or not course or not date or not class_name:
        return jsonify({
            "success": False,
            "message": "Data tidak lengkap"
        })

    conn = get_db()
    c = conn.cursor()

    #Hapus baris berdasarkan semua field termasuk class
    c.execute("""
        DELETE FROM attendance
        WHERE name=? AND course=? AND class=? AND datetime LIKE ?
    """, (name, course, class_name, f"{date}%"))

    conn.commit()
    conn.close()

    #Lokasi folder FOTO 
    photo_path = os.path.join(ATT_PHOTO_DIR, date, course, class_name, name)

    if os.path.exists(photo_path):
        shutil.rmtree(photo_path)

    return jsonify({
        "success": True,
        "message": f"Kehadiran {name} berhasil dihapus"
    })



@app.route('/download/<course>/<date>/<class_name>')
def download_csv_date(course, date, class_name):
    import csv
    path = f"db/{course}_{class_name}_{date}_attendance.csv"

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT name, course, class, datetime 
        FROM attendance 
        WHERE course=? AND class=? AND datetime LIKE ?
        ORDER BY datetime DESC
    """, (course, class_name, f"{date}%"))

    rows = c.fetchall()
    conn.close()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Nama", "Mata Kuliah", "Kelas", "Tanggal & Waktu"])
        writer.writerows(rows)

    return send_file(path, as_attachment=True)


# ------------------ PRESENSI BUTTON ------------------
@app.route('/rekam_presensi', methods=['POST'])
def rekam_presensi():
    global last_face_names
    data = request.get_json()

    course = data.get("course")
    class_name = data.get("class")

    if not course or not class_name:
        return jsonify({"success": False, "messages": ["Isi kelas & mata kuliah dulu"]})

    conn = get_db()
    c = conn.cursor()
    pesan = []

    for (_, _, _, _, name) in last_face_names:
        if name != "Unknown":
            today = datetime.now().strftime("%Y-%m-%d")

            c.execute("""
                SELECT * FROM attendance 
                WHERE name=? AND course=? AND class=? AND DATE(datetime)=?
            """, (name, course, class_name, today))

            already = c.fetchone()

            if not already:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("""
                    INSERT INTO attendance (name, course, class, datetime)
                    VALUES (?, ?, ?, ?)
                """, (name, course, class_name, now))

                pesan.append(f"{name} berhasil absen {course} ({class_name}) pada {now}")

    conn.commit()
    conn.close()

    if pesan:
        return jsonify({"success": True, "messages": pesan})
    else:
        return jsonify({
          "success": False,
          "messages": ["Tidak ada wajah baru terdeteksi / sudah absen hari ini"]
        })

# ------------------ MANAGE ------------------
@app.route('/manage')
def manage():
    """Halaman untuk manajemen data wajah"""
    persons = []
    if os.path.exists(FACE_DIR):
        persons = [d for d in os.listdir(FACE_DIR) if os.path.isdir(os.path.join(FACE_DIR, d))]
    return render_template("manage.html", persons=persons)

@app.route('/delete_person', methods=['POST'])
def delete_person():
    """Hapus data wajah & embeddings seseorang"""
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
    """Tambah wajah baru dari kamera (base64 image dari client)"""
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

        # decode base64 -> image
        img_str = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        cv_img = np.array(img)

        # deteksi wajah & crop
        detections = DeepFace.extract_faces(
            img_path=cv_img,
            detector_backend="ssd",
            enforce_detection=False
        )

        if detections:
            det = detections[0]
            face_crop = det["face"]
            face_crop = (face_crop * 255).astype("uint8")  # normalisasi
            face_crop = cv2.resize(face_crop, (160, 160))
            cv2.imwrite(os.path.join(save_dir, filename), face_crop)
        else:
            img.save(os.path.join(save_dir, filename))

        # update embeddings
        emb = DeepFace.represent(
            img_path=os.path.join(save_dir, filename),
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False
        )[0]["embedding"]

        if name not in face_embeddings:
            face_embeddings[name] = []
        face_embeddings[name].append(np.array(emb))
        save_embeddings()

        return jsonify({"success": True, "count": existing+1}), 200
    except Exception as e:
        print("Error add_face_cam:", e)
        return "Gagal menyimpan wajah", 500

@app.route("/add_face_upload", methods=["POST"])
def add_face_upload():
    """
    Tambah dataset wajah dari file upload
    """
    name = request.form.get("name")
    files = request.files.getlist("files[]")

    if not name or not files:
        return jsonify({
            "success": False,
            "message": "Nama dan file wajib diisi"
        })

    save_dir = os.path.join(FACE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    failed = 0

    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
            img_np = np.array(img)

            detections = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend="ssd",
                enforce_detection=False
            )

            if not detections:
                failed += 1
                continue

            det = detections[0]
            face_crop = det["face"]
            face_crop = (face_crop * 255).astype("uint8")

            face_crop = cv2.resize(face_crop, (160, 160))

            existing = len([
                x for x in os.listdir(save_dir)
                if x.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            filename = f"{name}_{existing+1}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), face_crop)

            # ---------- update embeddings ----------
            emb = DeepFace.represent(
                img_path=os.path.join(save_dir, filename),
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False
            )[0]["embedding"]

            if name not in face_embeddings:
                face_embeddings[name] = []

            face_embeddings[name].append(np.array(emb))
            save_embeddings()

            saved += 1

        except Exception as e:
            print("UPLOAD ERROR:", e)
            failed += 1

    return jsonify({
        "success": True,
        "message": f"Upload selesai. Berhasil: {saved}, Gagal: {failed}"
    })

@app.route("/save_dataset", methods=["POST"])
def save_dataset():
    data = request.get_json()
    name = data.get("name")
    images = data.get("images", [])

    if not name or not images:
        return jsonify({"success": False, "message": "Data tidak lengkap"})

    save_dir = os.path.join(FACE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    new_embs = []

    for i, img_data in enumerate(images):
        try:
            img_str = re.sub('^data:image/.+;base64,', '', img_data)
            img = Image.open(BytesIO(base64.b64decode(img_str))).convert("RGB")
            img_np = np.array(img)

            # deteksi wajah
            dets = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend="ssd",
                enforce_detection=False
            )

            if not dets:
                continue

            face = dets[0]["face"]
            face = (face * 255).astype("uint8")
            face = cv2.resize(face, (160,160))

            filename = os.path.join(save_dir, f"{name}_{i+1}.jpg")
            cv2.imwrite(filename, face)

            emb = DeepFace.represent(
                img_path=face,
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False
            )[0]["embedding"]

            new_embs.append(np.array(emb))

        except Exception as e:
            print("SAVE DATASET ERROR:", e)

    if not new_embs:
        return jsonify({"success": False, "message": "Tidak ada wajah valid"})

    # update embeddings
    if name not in face_embeddings:
        face_embeddings[name] = []

    face_embeddings[name].extend(new_embs)
    save_embeddings()

    return jsonify({"success": True, "message": f"Dataset untuk {name} berhasil disimpan ({len(new_embs)} foto valid)"})


@app.route("/capture")
def capture():
    return render_template("capture.html")

@app.route("/classify_faces", methods=["POST"])
def classify_faces():
    data = request.get_json()

    date = data.get("date")
    course = data.get("course")
    images = data.get("images")

    if not date or not course or not images:
        return jsonify({
            "success": False,
            "message": "Data tidak lengkap"
        })

    # ----------------------------------
    # SIMPAN HASIL VOTING
    # { name: [score1, score2, ...] }
    # ----------------------------------
    vote_results = {}

    for image_data in images:
        try:
            # Decode base64 -> numpy image
            img_str = re.sub('^data:image/.+;base64,', '', image_data)
            img_bytes = base64.b64decode(img_str)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            # Deteksi wajah
            detections = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend="ssd",
                enforce_detection=False
            )

            for det in detections:
                face_img = det["face"]

                # Embedding wajah
                emb = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    detector_backend="skip",
                    enforce_detection=False
                )[0]["embedding"]

                # Matching ke database
                best_name = None
                best_score = -1

                for name, emb_list in face_embeddings.items():
                    for ref_emb in emb_list:
                        score = cosine_similarity([emb], [ref_emb])[0][0]
                        if score > best_score:
                            best_score = score
                            best_name = name

                # Threshold
                if best_score >= 0.75:
                    if best_name not in vote_results:
                        vote_results[best_name] = []
                    vote_results[best_name].append(best_score)

        except Exception as e:
            print("Image processing error:", e)

    if not vote_results:
        return jsonify({
            "success": False,
            "message": "Tidak ada wajah yang dikenali"
        })

    # ----------------------------------
    # FINAL RESULT (VOTING)
    # ----------------------------------
    final_names = []
    for name, scores in vote_results.items():
        avg_score = sum(scores) / len(scores)
        if avg_score >= 0.8:   # threshold voting akhir
            final_names.append(name)

    if not final_names:
        return jsonify({
            "success": False,
            "message": "Wajah terdeteksi, tapi tidak cukup yakin"
        })

    # ----------------------------------
    # SIMPAN KE DATABASE
    # ----------------------------------
    conn = get_db()
    c = conn.cursor()
    inserted = []

    for name in final_names:
        # Cek apakah sudah absen di tanggal tsb
        c.execute("""
            SELECT id FROM attendance
            WHERE name=? AND course=? AND DATE(datetime)=?
        """, (name, course, date))

        if not c.fetchone():
            now = f"{date} {datetime.now().strftime('%H:%M:%S')}"
            c.execute("""
                INSERT INTO attendance (name, course, datetime)
                VALUES (?, ?, ?)
            """, (name, course, now))
            inserted.append(name)

    conn.commit()
    conn.close()

    if not inserted:
        return jsonify({
            "success": False,
            "message": "Semua wajah sudah tercatat sebelumnya"
        })

    return jsonify({
    "success": True,
    "message": f"Absensi berhasil: {', '.join(inserted)}",
    "results": [
        {
            "name": name,
            "confidence": round(sum(vote_results[name]) / len(vote_results[name]), 3)
        }
        for name in final_names
    ]
})

@app.route("/preview_faces", methods=["POST"])
def preview_faces():
    data = request.get_json()
    images = data.get("images")

    vote_results = {}
    annotated_images = []

    for image_data in images:
        img_str = re.sub('^data:image/.+;base64,', '', image_data)
        img = Image.open(BytesIO(base64.b64decode(img_str))).convert("RGB")
        img_np = np.array(img)

        detections = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend="ssd",
            enforce_detection=False
        )

        for det in detections:
            fa = det["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            face = det["face"]

            emb = DeepFace.represent(
                img_path=face,
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False
            )[0]["embedding"]

            best_name, best_score = "Unknown", -1
            for name, emb_list in face_embeddings.items():
                for ref in emb_list:
                    score = cosine_similarity([emb], [ref])[0][0]
                    if score > best_score:
                        best_name, best_score = name, score

            if best_score >= 0.75:
                vote_results.setdefault(best_name, []).append(best_score)

            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_np, best_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, buf = cv2.imencode(".jpg", img_np)
        annotated_images.append(
            "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        )

    results = [
        {
            "name": n,
            "confidence": round(sum(s)/len(s), 3)
        }
        for n, s in vote_results.items()
        if sum(s)/len(s) >= 0.8
    ]

    return jsonify({
        "success": True,
        "results": results,
        "annotated_images": annotated_images
    })

@app.route("/save_attendance", methods=["POST"])
def save_attendance():
    data = request.get_json()
    date = data.get("date")
    course = data.get("course")
    images = data.get("images", [])
    class_name = data.get("class")
    results = data.get("results", [])

    if not date or not course or not class_name or not results:
        return jsonify({
            "success": False,
            "message": "Lengkapi tanggal, mata kuliah, kelas, dan preview dulu"
    })


    conn = get_db()
    c = conn.cursor()
    inserted = []

    try:
        for r in results:
            name = r["name"]

            c.execute("""
                SELECT id FROM attendance
                WHERE name=? AND course=? AND class=? AND DATE(datetime)=?
            """, (name, course, class_name, date))


            if c.fetchone():
                continue

            now = f"{date} {datetime.now().strftime('%H:%M:%S')}"
            c.execute("""
                INSERT INTO attendance (name, course, class, datetime)
                VALUES (?, ?, ?, ?)
            """, (name, course, class_name, now))

            inserted.append(name)

            save_dir = os.path.join(ATT_PHOTO_DIR, date, course, class_name, name)
            os.makedirs(save_dir, exist_ok=True)

            for i, img_data in enumerate(images):
                img_str = re.sub('^data:image/.+;base64,', '', img_data)
                img = Image.open(BytesIO(base64.b64decode(img_str)))
                img.save(os.path.join(save_dir, f"{i+1}.jpg"))

        conn.commit()

    except Exception as e:
        conn.rollback()
        print("SAVE ATTENDANCE ERROR:", e)
        return jsonify({
            "success": False,
            "message": "Gagal menyimpan absensi"
        }), 500

    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": f"Absensi disimpan: {', '.join(inserted)}"
    })



# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=False)
  # jalankan server Flask
