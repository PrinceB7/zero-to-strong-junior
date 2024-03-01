import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("weights/yolov8m-seg.pt")
names = model.model.names

input_video_path = "videos/football.mp4"
output_video_path = f"out/segment_{os.path.basename(input_video_path)}"

w,h = 1280, 720
# videoni o'qish uchun
cap = cv2.VideoCapture(input_video_path)
# videoni saqlash uchun
out = cv2.VideoWriter(output_video_path, 0x00000021,
                      30, (w, h))

while True:
    ret, im0 = cap.read() # im0 = frame
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = cv2.resize(im0, (w, h))

    # inference / predict
    results = model.predict(im0)
    
    # decoration
    clss = results[0].boxes.cls.cpu().tolist()
    masks = results[0].masks.xy

    annotator = Annotator(im0, line_width=2)

    # filter qilish uchun
    for mask, cls in zip(masks, clss):
        if names[int(cls)] in ['person', 'car', 'truck']:
            try:
                annotator.seg_bbox(mask=mask,
                           mask_color=colors(int(cls), True),
                           det_label=names[int(cls)])
            except: continue

    out.write(im0)
    cv2.imshow("instance-segmentation", cv2.resize(im0, (1280,720)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()