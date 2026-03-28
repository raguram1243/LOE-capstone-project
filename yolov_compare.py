from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load models
model_v8 = YOLO("A:/PROJECT/Line marking/runs/detect/yolov8_runs/train_exp1/weights/best.pt")
model_v9 = YOLO("A:/PROJECT/Line marking/runs/detect/yolov9_runs/train_exp1/weights/best.pt")

# Image path
image_path = "A:/PROJECT/Line marking/yolo_dataset/images/train/000111.jpg"

# Predictions
results_v8 = model_v8(image_path)
results_v9 = model_v9(image_path)

# Plot
img_v8 = results_v8[0].plot()
img_v9 = results_v9[0].plot()

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("YOLOv8")
plt.imshow(cv2.cvtColor(img_v8, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("YOLOv9")
plt.imshow(cv2.cvtColor(img_v9, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

# ===============================
# 🔹 VALIDATION METRICS (DATASET)
# ===============================
metrics_v8 = model_v8.val(data=r"A:\PROJECT\Line marking\dataset.yaml", verbose=False)
metrics_v9 = model_v9.val(data=r"A:\PROJECT\Line marking\dataset.yaml", verbose=False)

# Extract metrics
v8_map50 = metrics_v8.box.map50
v8_map95 = metrics_v8.box.map
v8_precision = metrics_v8.box.mp
v8_recall = metrics_v8.box.mr

v9_map50 = metrics_v9.box.map50
v9_map95 = metrics_v9.box.map
v9_precision = metrics_v9.box.mp
v9_recall = metrics_v9.box.mr

# ===============================
# 🔹 IMAGE PREDICTION CONFIDENCE
# ===============================

res_v8 = model_v8(image_path)
res_v9 = model_v9(image_path)

def get_avg_conf(res):
    if res[0].boxes is None:
        return 0
    return res[0].boxes.conf.mean().item()

v8_conf = get_avg_conf(res_v8)
v9_conf = get_avg_conf(res_v9)

# ===============================
# 📊 BAR CHART
# ===============================
labels = ["mAP50", "mAP50-95", "Precision", "Recall", "Confidence"]

v8_values = [v8_map50, v8_map95, v8_precision, v8_recall, v8_conf]
v9_values = [v9_map50, v9_map95, v9_precision, v9_recall, v9_conf]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(x - width/2, v8_values, width, label="YOLOv8")
plt.bar(x + width/2, v9_values, width, label="YOLOv9")

plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("YOLOv8 vs YOLOv9 Performance Comparison")
plt.xticks(x, labels)
plt.legend()

# Add value labels
for i in range(len(labels)):
    plt.text(x[i] - width/2, v8_values[i], f"{v8_values[i]:.2f}", ha='center')
    plt.text(x[i] + width/2, v9_values[i], f"{v9_values[i]:.2f}", ha='center')

plt.tight_layout()
plt.show()