# ============================
# Image Object Partition using YOLOv8
# Author: Shishir + ChatGPT
# ============================

# STEP 1: Import libraries
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# STEP 2: Select image from file
root = Tk()
root.withdraw()  # Hide main tkinter window
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected. Exiting...")
    exit()

# STEP 3: Load pre-trained YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# STEP 4: Perform detection
print("Detecting objects...")
results = model(image_path)

# STEP 5: Read the original image
img = cv2.imread(image_path)

# STEP 6: Process detected boxes
boxes = results[0].boxes.xyxy  # bounding box coordinates
names = results[0].names       # class names (e.g. dog, cat)
classes = results[0].boxes.cls # class IDs

if len(boxes) == 0:
    print("No objects detected!")
    exit()

# STEP 7: Crop and show each detected object
for i, (box, cls) in enumerate(zip(boxes, classes)):
    x1, y1, x2, y2 = map(int, box)
    cropped = img[y1:y2, x1:x2]
    label = names[int(cls)]
    cv2.imshow(f"Object {i+1}: {label}", cropped)

print(f"✅ {len(boxes)} objects detected and displayed.")
cv2.waitKey(0)
cv2.destroyAllWindows()
