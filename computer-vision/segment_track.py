import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
track_history = defaultdict(lambda: [])


model = YOLO("weights/yolov8m-seg.pt")
names = model.model.names

input_video_path = "./videos/mall_front.mp4"
output_video_path = f"out/segment_{os.path.basename(input_video_path)}"


w, h = 1280, 720
cap = cv2.VideoCapture(input_video_path)
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30, (w, h))


while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = cv2.resize(im0, (w, h))
    
    # predict va tracking
    results = model.track(im0, persist=True)
    
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotator = Annotator(im0, line_width=2)

    for mask, track_id in zip(masks, track_ids):
        annotator.seg_bbox(mask=mask,
                           mask_color=colors(track_id, True),
                           track_label=str(track_id))

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()