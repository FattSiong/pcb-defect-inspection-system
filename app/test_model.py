from ultralytics import YOLO

model = YOLO(
    "app/models/technique_d_best.pt"
)

print("Model loaded successfully.")
print("Classes:")
print(model.names)