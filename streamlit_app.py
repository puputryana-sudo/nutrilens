import streamlit as st
import cv2
from PIL import Image
import supervision as sv
import numpy as np
import roboflow
import pandas as pd
import os

# path relatif supaya aman saat deploy
BASE_DIR = os.path.dirname(__file__)
rules_path = os.path.join(BASE_DIR, "rule_diabetes_python.xlsx")
rules_df = pd.read_excel(rules_path)

# untuk sistem rekomendasi
def generate_recommendations(detections):
    recs = []
    counts = pd.Series(detections).value_counts()

    for cls, count in counts.items():
        rule = rules_df[rules_df['class'] == cls]
        if not rule.empty:
            max_allowed = int(rule['max_allowed'].values[0])
            if count < max_allowed:
                recommendation = rule['less_than'].values[0]
            elif count == max_allowed:
                recommendation = rule['equal'].values[0]
            else:
                recommendation = rule['greater_than'].values[0]
            recs.append(f"{cls}: {recommendation}")
    return recs


# Fungsi get_model yang sudah diperbaiki untuk API Roboflow
@st.cache_resource
def get_model(api_key):
    """
    Memuat model dari Roboflow dan menyimpannya dalam cache.
    """
    try:
        rf = roboflow.Roboflow(api_key=api_key)
        
        # Menggunakan project ID dan versi secara langsung
        project = rf.workspace("putriana-dwi-agustin-ayet0").project("nutrilens-qutk4")
        version = project.version(7)
        model = version.model
        
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari Roboflow. Pastikan API key dan project ID benar. Error: {e}")
        return None

# Judul aplikasi
st.title("Deteksi Nutrisi pada Sajian Piring")

# Dua opsi input: file uploader dan kamera
uploaded_file = st.file_uploader("Pilih gambar sajian", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Ambil gambar dengan kamera")

# Mengambil file yang diunggah atau diambil dari kamera
input_image = uploaded_file if uploaded_file else camera_image

if input_image is not None:
    # baca gambar dengan PIL
    pil_img = Image.open(input_image).convert("RGB")
    
    # tampilkan gambar asli
    st.image(pil_img, caption="Gambar asli", use_container_width=True)

    # load model roboflow
    model = get_model(api_key="RoWNb7wk6nYlQZYojZVY")

    if model is not None:
        try:
            # konversi PIL Image ke array NumPy
            image_array = np.array(pil_img)
            
            # infer pakai NumPy array
            result = model.predict(image_array, confidence=15, overlap=50).json()

            # konversi hasil ke format detections supervision
            detections = sv.Detections.from_inference(result)

            # anotasi hasil deteksi
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # konversi array RGB ke BGR untuk OpenCV
            img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            annotated = box_annotator.annotate(scene=img_bgr, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections)

            # convert hasil anotasi ke RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Hasil Deteksi", use_container_width=True)

            # konversi hasil ke list nama kelas
            detected_classes = [pred["class"] for pred in result["predictions"]]

            # generate rekomendasi
            recommendations = generate_recommendations(detected_classes)

            # tampilkan rekomendasi di Streamlit
            st.markdown("### Rekomendasi Gizi:")
            for rec in recommendations:
                st.write(f"- {rec}")

           
        except Exception as e:
            st.error(f"Terjadi kesalahan saat inferensi atau anotasi. Error: {e}")
