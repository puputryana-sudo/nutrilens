import streamlit as st
import cv2
from PIL import Image
from inference import get_model
import supervision as sv
import numpy as np

# Judul aplikasi
st.title("Deteksi Nutrisi pada Sajian Piring")

# Upload gambar dari user
uploaded_file = st.file_uploader("Pilih gambar sajian", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # baca gambar dengan PIL
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # tampilkan gambar asli
    st.image(pil_img, caption="Gambar asli", use_column_width=True)

    # load model roboflow (bisa dimasukkan di awal supaya tidak load berulang)
    model = get_model("nutrilens-qutk4/7", api_key="RoWNb7wk6nYlQZYojZVY")

    # infer pakai PIL image → ambil hasil pertama [0]
    result = model.infer(pil_img)[0]

    # convert hasil ke objek detections supervision
    detections = sv.Detections.from_inference(result)

    # anotasi hasil deteksi
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # konversi PIL ke array (OpenCV format)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Supervison pakai BGR

    annotated = box_annotator.annotate(scene=img_bgr, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    # convert hasil annotasi ke RGB agar bisa ditampilkan di Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

    # cetak output hasil deteksi
    st.write("Hasil prediksi (class & confidence):")
    st.json(result['predictions'])
