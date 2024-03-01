from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('weights/yolov8m.pt')  # load a custom model

# Predict with the model
input = "../5. neural_network/datasets/cats-dogs/test/cats/10.jpg"
output = model("inputs/27.jpg", save=True)  # predict on an image

print(output[0])

'''

input -> model -> output

video -> frames -> model (inference) -> frames -> video
'''


from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train/weights/best.pt') # best vs last

# Train the model with 1 GPUs
results = model.predict(data='coco128.yaml', epochs=100, imgsz=640, device=[0], verbose=False)