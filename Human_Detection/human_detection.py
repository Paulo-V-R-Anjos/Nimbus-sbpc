import os
from ultralytics import YOLO
import cv2
import schedule
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1) Determine base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2) Use the existing temp/images folder under the script directory
out_dir = os.path.join(BASE_DIR, "temp", "images")
os.makedirs(out_dir, exist_ok=True)

# 3) Pick device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 4) Load YOLO model
print("Loading pretrained YOLO…")
model = YOLO("yolov8x.pt").to(device)

# 5) Open webcam (index 0 or change as needed)
print("Loading cam…")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 6) Prepare live-view window
cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)

# 7) Scheduling parameters
capture_interval = 1    # seconds between captures
publish_interval = 60   # seconds to average over
img_counter_max = publish_interval // capture_interval
predicted_counts = []

# 8) Load a bold font (fall back to default)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=50)
except IOError:
    font = ImageFont.load_default()
orange = (255, 165,  80)  # soft orange RGB

def capture_and_predict():
    global predicted_counts

    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame")
        return

    # BGR → RGB for PIL & model
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # Inference
    results = model(rgb, classes=0, conf=0.40)

    # Draw thicker boxes & count humans
    human_count = 0
    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cls in zip(boxes, classes):
            if cls == 0:
                human_count += 1
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=orange,
                    width=6  # even thicker border
                )

    predicted_counts.append(human_count)
    print(f"Captured frame, detected {human_count} person(s)")

    # Draw the current count with white fill and black outline
    text = f"Número de Pessoas: {human_count}"
    draw.text(
        (10, 10),
        text,
        font=font,
        fill="white",
        stroke_width=2,
        stroke_fill="black"
    )

    # Convert back to BGR for saving & display
    annotated_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Save (overwrite) the same filename each time
    out_path = os.path.join(out_dir, "image_with_count.png")
    success = cv2.imwrite(out_path, annotated_bgr)
    if success:
        print("Wrote annotated image to:", out_path)
    else:
        print("Failed to save image to:", out_path)

    # Show live in a window
    cv2.imshow("Live Detection", annotated_bgr)
    cv2.waitKey(1)  # allow GUI to refresh

    # After enough frames, compute & log the mean
    if len(predicted_counts) >= img_counter_max:
        mean_count = round(sum(predicted_counts) / len(predicted_counts))
        predicted_counts.clear()
        print(f"Mean over last {img_counter_max} frames: {mean_count}")

# 9) Start scheduler
schedule.every(capture_interval).seconds.do(capture_and_predict)
print("Starting prediction loop…")
try:
    while True:
        schedule.run_pending()
finally:
    cam.release()
    cv2.destroyAllWindows()
