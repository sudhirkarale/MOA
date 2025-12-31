import os, sys, io, time
import cv2
import onnxruntime as ort
import numpy as np
from collections import Counter
from contextlib import redirect_stderr

# -------------------------------
# Silence ORT logs
# -------------------------------
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"
os.environ["ORT_LOGGING_LEVEL"] = "ERROR"
os.environ["ORT_DISABLE_ALL_LOGGING"] = "1"


def silent_ort_session(onnx_path, providers=None):
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.log_verbosity_level = 0
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = providers or ["CPUExecutionProvider"]
    sink = io.StringIO()
    with redirect_stderr(sink):
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


def softmax(x):
    x = np.atleast_2d(x)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def mode_stable(arr):
    c = Counter(arr)
    m = max(c.values())
    return min(k for k, v in c.items() if v == m)


# -------------------------------
# Config
# -------------------------------
TARGET_FPS = 5.0
WINDOW_FRAMES = 25
SAMPLE_INTERVAL = 1.0 / TARGET_FPS

# Model paths
model_path_emo = "/home/sudhir/Desktop/OCT_23/Mood_cls/facial_expression_recognition_mobilefacenet_2022july.onnx"
model_path_face = "/home/sudhir/Desktop/Mood_Detection/face_detection_yunet_2023mar.onnx"
model_path_gender = "/home/sudhir/Desktop/OCT_23/Gender_classification/GenderClassification/YOLOv11_gender4/weights/best.onnx"
model_path_age = "/home/sudhir/Desktop/OCT_23/Age/ageClassification/YOLOv11s_age_2245/weights/best.onnx"

# Labels
mood_classes = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
gender_classes = ["female", "male"]
age_classes = ["18-24", "Below 18", "25-34", "35-44", "45-54", "55-80"]


# -------------------------------
# Alignment template
# -------------------------------
STD_LM5 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def _similarity_transform_nonreflective(src, dst):
    x = dst[:, 0].reshape((-1, 1))
    y = dst[:, 1].reshape((-1, 1))

    X = np.vstack((
        np.hstack((x, y, np.ones((len(dst), 1)), np.zeros((len(dst), 1)))),
        np.hstack((y, -x, np.zeros((len(dst), 1)), np.ones((len(dst), 1))))
    ))

    u = src[:, 0].reshape((-1, 1))
    v = src[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    r, _, _, _ = np.linalg.lstsq(X, U, rcond=-1)
    r = np.squeeze(r)

    sc, ss, tx, ty = r
    Tinv = np.array([[sc, -ss, 0],
                     [ss,  sc, 0],
                     [tx,  ty, 1]], dtype=np.float32)

    T = np.linalg.inv(Tinv)
    T[:, 2] = np.array([0, 0, 1], dtype=np.float32)
    return T


def align_112_bgr(image_bgr, lm5):
    T = _similarity_transform_nonreflective(lm5.astype(np.float32), STD_LM5)
    M = T[:, 0:2].T
    return cv2.warpAffine(image_bgr, M, (112, 112))


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_mood(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (112, 112))
    face = face_rgb.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return np.expand_dims(face, axis=0)


def preprocess_gender(face_bgr):
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def preprocess_age(face_bgr):
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


# -------------------------------
# Load models
# -------------------------------
emo_sess = silent_ort_session(model_path_emo)
emo_in = emo_sess.get_inputs()[0].name
emo_out = emo_sess.get_outputs()[0].name

gender_sess = silent_ort_session(model_path_gender)
gender_in = gender_sess.get_inputs()[0].name
gender_out = gender_sess.get_outputs()[0].name

age_sess = silent_ort_session(model_path_age)
age_in = age_sess.get_inputs()[0].name
age_out = age_sess.get_outputs()[0].name


# -------------------------------
# Face detector
# -------------------------------
detector = cv2.FaceDetectorYN.create(
    model=model_path_face,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)


# -------------------------------
# Capture
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit(0)

# Warm frames
for _ in range(10):
    ok, _ = cap.read()
    if ok:
        break

prev_size = None

mood_buf, gender_buf, age_buf = [], [], []
last_sample_time = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        size = (w, h)

        if size != prev_size:
            detector.setInputSize(size)
            prev_size = size

        now = time.time()
        if now - last_sample_time < SAMPLE_INTERVAL:
            time.sleep(0.001)
            continue
        last_sample_time = now

        faces_tuple = detector.detect(frame)
        faces = None if faces_tuple is None else faces_tuple[1]

        if faces is None or len(faces) == 0:
            continue

        # pick largest face
        idx = int(np.argmax([(int(f[2]) * int(f[3])) for f in faces]))
        f = faces[idx]

        x, y, w_f, h_f = map(int, f[:4])
        x = max(0, x)
        y = max(0, y)
        w_f = max(0, min(w_f, w - x))
        h_f = max(0, min(h_f, h - y))

        if w_f < 30 or h_f < 30:
            continue

        lm = None
        score = f[-1] if f.shape[0] >= 16 else 1.0

        if score >= 0.7 and f.shape[0] >= 15:
            try:
                lm = f[4:14].reshape(5, 2).astype(np.float32)
                lm[:, 0] = np.clip(lm[:, 0], 0, w - 1)
                lm[:, 1] = np.clip(lm[:, 1], 0, h - 1)
            except:
                lm = None

        face_crop = frame[y:y + h_f, x:x + w_f]

        # mood
        if lm is not None:
            face_mood = align_112_bgr(frame, lm)
        else:
            face_mood = face_crop

        emo_probs = softmax(
            emo_sess.run([emo_out], {emo_in: preprocess_mood(face_mood)})[0]
        )
        mood_buf.append(int(np.argmax(emo_probs)))

        # gender
        g_out = gender_sess.run([gender_out], {gender_in: preprocess_gender(face_crop)})[0]
        g_out = np.atleast_2d(g_out)
        gender_buf.append(int(np.argmax(softmax(g_out))))

        # age
        a_out = age_sess.run([age_out], {age_in: preprocess_age(face_crop)})[0]
        a_out = np.atleast_2d(a_out)
        age_buf.append(int(np.argmax(softmax(a_out))))

        # stop condition
        if len(mood_buf) >= WINDOW_FRAMES:
            mood_mode = mode_stable(mood_buf)
            gender_mode = mode_stable(gender_buf)
            age_mode = mode_stable(age_buf)

            print(f"{mood_classes[mood_mode]} {gender_classes[gender_mode]} {age_classes[age_mode]}")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
