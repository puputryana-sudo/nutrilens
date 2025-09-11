import streamlit as st
import cv2
from PIL import Image
import supervision as sv
import numpy as np
import roboflow

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
        model = version.version_model
        
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari Roboflow. Pastikan API key dan project ID benar. Error: {e}")
        return None

# Judul aplikasi
st.title("Deteksi Nutrisi pada Sajian Piring")

# Upload gambar dari user
uploaded_file = st.file_uploader("Pilih gambar sajian", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # baca gambar dengan PIL
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # tampilkan gambar asli
    st.image(pil_img, caption="Gambar asli", use_column_width=True)

    # load model roboflow
    model = get_model(api_key="RoWNb7wk6nYlQZYojZVY")

    if model is not None:
        try:
            # konversi PIL Image ke array NumPy
            image_array = np.array(pil_img)
            
            # infer pakai NumPy array
            result = model.predict(image_array, confidence=40, overlap=30).json()

            # konversi hasil ke format detections supervision
            detections = sv.Detections.from_roboflow(result)

            # anotasi hasil deteksi
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # konversi array RGB ke BGR untuk OpenCV
            img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            annotated = box_annotator.annotate(scene=img_bgr, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections)

            # convert hasil anotasi ke RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

            # cetak output hasil deteksi
            st.write("Hasil prediksi (class & confidence):")
            st.json(result['predictions'])
        except Exception as e:
            st.error(f"Terjadi kesalahan saat inferensi atau anotasi. Error: {e}")
