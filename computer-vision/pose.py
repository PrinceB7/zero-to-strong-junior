from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolov8m-pose.pt')  # load an official model

# Predict with the model
results = model('videos/football.mp4', show=True, save=True, conf=0.7)  # predict on an image