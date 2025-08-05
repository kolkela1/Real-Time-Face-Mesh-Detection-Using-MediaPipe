import cv2
import mediapipe as mp

# تهيئة أدوات MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# لرسم النقاط
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # تحويل الألوان من BGR إلى RGB (مطلوب لـ MediaPipe)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # تحليل الوجه
    results = face_mesh.process(img_rgb)

    # لو اتعرف على وجه
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # رسم النقاط والخطوط
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # الشبكة
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    cv2.imshow('Face Mesh Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # اضغط Esc للخروج
        break

cap.release()
cv2.destroyAllWindows()
