import cv2
from PIL import Image
from inference import get_model
import supervision as sv

# load model roboflow
model = get_model("nutrilens-qutk4/7", api_key="RoWNb7wk6nYlQZYojZVY")

# baca pakai cv2
img = cv2.imread("contoh.jpg")

# convert dari BGR (cv2) ke RGB (PIL)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

# infer pakai PIL image → ambil hasil pertama [0]
result = model.infer(pil_img)[0]

# convert hasil ke objek detections supervision
detections = sv.Detections.from_inference(result)

# anotasi hasil deteksi
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated = box_annotator.annotate(scene=img, detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections)

# tampilkan hasil
sv.plot_image(annotated)

# cetak output hasil deteksi (misal: class & confidence)
print(result['predictions'])
