import streamlit as st
import cv2
from PIL import Image
import supervision as sv
import numpy as np
from ultralytics import YOLO

# Menggunakan cache_resource untuk menyimpan model
@st.cache_resource
def load_yolo_model():
    # Ganti dengan path ke model YOLO Anda
    # Anda bisa mengunduhnya dari Roboflow dan menyimpannya di folder yang sama
    # Contoh: 'best.pt'
    model_path = 'best.pt' 
    return YOLO(model_path)

# Judul aplikasi
st.title("Deteksi Nutrisi pada Sajian Piring")

# Upload gambar dari user
uploaded_file = st.file_uploader("Pilih gambar sajian", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # baca gambar dengan PIL
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # tampilkan gambar asli
    st.image(pil_img, caption="Gambar asli", use_column_width=True)

    try:
        # load model YOLO dari file lokal
        model = load_yolo_model()
        
        # infer pakai PIL image
        result = model(pil_img)[0]

        # konversi hasil inferensi ke format supervision
        detections = sv.Detections.from_ultralytics(result)

        # anotasi hasil deteksi
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # konversi PIL ke array (OpenCV format)
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        annotated = box_annotator.annotate(scene=img_bgr, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)

        # convert hasil anotasi ke RGB
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

        # cetak output hasil deteksi
        st.write("Hasil prediksi (class & confidence):")
        
        predictions = []
        for det in detections:
            predictions.append({
                "class_id": int(det[3]),
                "class_name": result.names[int(det[3])],
                "confidence": float(det[2]),
                "box": [float(i) for i in det[0]]
            })
        st.json(predictions)

    except Exception as e:
        st.error(f"Terjadi kesalahan. Pastikan model YOLO berada di folder yang sama dengan nama `best.pt`. Error: {e}")
