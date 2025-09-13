import streamlit as st
import cv2
from PIL import Image
import supervision as sv
import numpy as np
import roboflow
from collections import Counter

# Fungsi get_model yang sudah diperbaiki untuk API Roboflow
@st.cache_resource
def get_model(api_key):
    """
    Memuat model dari Roboflow dan menyimpannya dalam cache.
    """
    try:
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace("putriana-dwi-agustin-ayet0").project("nutrilens-qutk4")
        version = project.version(7)
        model = version.model
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari Roboflow. Pastikan API key dan project ID benar. Error: {e}")
        return None


# aturan untuk rekomendasi
rules = {
    "Karbohidrat": {
        "max_servings": 3,
        "max_energy_kcal": 450,
        "max_carbs_g": 90
    },
    "Protein hewani": {
        "max_servings": 2,
        "max_energy_kcal": 400,
        "max_protein_g": 45
    },
    "Protein nabati": {
        "max_servings": 3,
        "max_energy_kcal": 360,
        "max_protein_g": 30
    },
    "Sayur": {
        "recommendation": "Lebih banyak lebih baik"
    },
    "Pelengkap": {
        "max_servings": 2,
        "max_energy_kcal": 200,
        "max_carbs_g": 40
    }
}


# sistem rekomendasi
def check_recommendation(detected_classes):
    counts = Counter(detected_classes)
    messages = []

    for food_class, count in counts.items():
        if food_class in rules:
            rule = rules[food_class]

            # cek batas sajian
            if "max_servings" in rule and count > rule["max_servings"]:
                messages.append(
                    f"⚠️ {food_class}: {count} sajian (melebihi batas {rule['max_servings']})."
                )
            else:
                messages.append(
                    f"✅ {food_class}: {count} sajian masih aman."
                )

            # info gizi batas aman
            if "max_energy_kcal" in rule:
                messages.append(f"   ➡️ Energi ≤ {rule['max_energy_kcal']} kcal")
            if "max_carbs_g" in rule:
                messages.append(f"   ➡️ Karbohidrat ≤ {rule['max_carbs_g']} g")
            if "max_protein_g" in rule:
                messages.append(f"   ➡️ Protein ≤ {rule['max_protein_g']} g")
        else:
            messages.append(f"ℹ️ {food_class}: belum ada aturan spesifik.")

    return "\n".join(messages)


# Judul aplikasi
st.title("Deteksi Nutrisi pada Sajian Piring")

# Dua opsi input
uploaded_file = st.file_uploader("Pilih gambar sajian", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Ambil gambar dengan kamera")

input_image = uploaded_file if uploaded_file else camera_image

if input_image is not None:
    pil_img = Image.open(input_image).convert("RGB")
    st.image(pil_img, caption="Gambar asli", use_container_width=True)

    model = get_model(api_key="RoWNb7wk6nYlQZYojZVY")

    if model is not None:
        try:
            image_array = np.array(pil_img)
            result = model.predict(image_array, confidence=15, overlap=50).json()

            detections = sv.Detections.from_inference(result)

            # anotasi hasil deteksi
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            annotated = box_annotator.annotate(scene=img_bgr, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Hasil Deteksi", use_container_width=True)

            # ambil list kelas
            detected_classes = [pred["class"] for pred in result["predictions"]]

            if detected_classes:
                st.subheader("Rekomendasi Konsumsi")
                st.text(check_recommendation(detected_classes))
            else:
                st.info("Tidak ada makanan terdeteksi.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat inferensi atau anotasi. Error: {e}")
