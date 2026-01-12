# ============================
# YOLO11 Comprehensive Detection Dashboard
# Methods: Standard vs Tiled vs Hybrid
# ============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog

# --- Configuration ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
TILE_OVERLAP = 0.25  # 25% overlap

class LowLightEnhancer:
    """Handles CLAHE enhancement for low-light images."""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

def draw_boxes(image, boxes, classes, confs, names, color=(0, 255, 0)):
    """Draws boxes on a copy of the image."""
    img_copy = image.copy()
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {conf:.2f}"
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img_copy

def run_standard_inference(model, img):
    """Method A: Detects objects on the full image."""
    results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    boxes, classes, confs = [], [], []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        classes = results[0].boxes.cls.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()
    
    return boxes, classes, confs

def run_tiled_inference(model, img, tile_rows=2, tile_cols=2):
    """Method B: Detects objects on overlapping tiles."""
    h_img, w_img, _ = img.shape
    tile_h = int(h_img / tile_rows)
    tile_w = int(w_img / tile_cols)
    step_h = int(tile_h * (1 - TILE_OVERLAP))
    step_w = int(tile_w * (1 - TILE_OVERLAP))
    
    all_boxes, all_classes, all_confs = [], [], []
    
    for y in range(0, h_img, step_h):
        for x in range(0, w_img, step_w):
            # Calculate tile coordinates
            y2 = min(y + tile_h, h_img)
            x2 = min(x + tile_w, w_img)
            y1 = max(0, y2 - tile_h)
            x1 = max(0, x2 - tile_w)
            
            tile = img[y1:y2, x1:x2]
            results = model(tile, conf=CONF_THRESHOLD, verbose=False, agnostic_nms=True)
            
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                
                # Edge Artifact Filter
                bx1, by1, bx2, by2 = xyxy
                margin = 5
                touches_left = (bx1 < margin) and (x1 > 0)
                touches_right = (bx2 > (x2-x1)-margin) and (x2 < w_img)
                touches_top = (by1 < margin) and (y1 > 0)
                touches_bot = (by2 > (y2-y1)-margin) and (y2 < h_img)
                
                if touches_left or touches_right or touches_top or touches_bot:
                    continue 

                # Global coordinates
                global_box = [xyxy[0]+x1, xyxy[1]+y1, xyxy[2]+x1, xyxy[3]+y1]
                all_boxes.append(global_box)
                all_classes.append(cls)
                all_confs.append(conf_val)

    return all_boxes, all_classes, all_confs

def apply_nms(boxes, classes, confs):
    """Merges overlapping boxes using NMS."""
    if len(boxes) == 0:
        return [], [], []

    # Convert [x1, y1, x2, y2] to [x, y, w, h] for OpenCV
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([int(x1), int(y1), int(w), int(h)])
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh, 
        scores=confs, 
        score_threshold=CONF_THRESHOLD, 
        nms_threshold=NMS_THRESHOLD
    )
    
    final_boxes, final_classes, final_confs = [], [], []
    if len(indices) > 0:
        if isinstance(indices, tuple): indices = indices[0]
        indices = np.array(indices).flatten()
        
        for i in indices:
            idx = int(i)
            final_boxes.append(boxes[idx])
            final_classes.append(classes[idx])
            final_confs.append(confs[idx])
            
    return final_boxes, final_classes, final_confs

def main():
    # 1. Setup
    print("Initializing...")
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select Low Light Image")
    if not image_path: return

    print("Loading YOLO11n Model...")
    model = YOLO('yolo11n.pt') 

    # 2. Preprocessing
    print(f"Enhancing Image: {image_path}")
    original_bgr = cv2.imread(image_path)
    enhancer = LowLightEnhancer()
    img_bgr = enhancer.enhance(original_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Method A: Standard Detection
    print("Running Method A (Standard)...")
    boxes_std, cls_std, conf_std = run_standard_inference(model, img_bgr)

    # 4. Method B: Tiled Detection
    print("Running Method B (Tiled)...")
    boxes_tile, cls_tile, conf_tile = run_tiled_inference(model, img_bgr)
    # Clean Tiled results
    boxes_tile, cls_tile, conf_tile = apply_nms(boxes_tile, cls_tile, conf_tile)

    # 5. Method C: Hybrid (Standard + Tiled)
    print("Running Method C (Hybrid Merge)...")
    boxes_hyb = boxes_std + boxes_tile
    cls_hyb = cls_std + cls_tile
    conf_hyb = conf_std + conf_tile
    
    boxes_hyb, cls_hyb, conf_hyb = apply_nms(boxes_hyb, cls_hyb, conf_hyb)

    # 6. Prepare Stats Data
    # Calculate stats handling empty lists
    def get_stats(confs):
        if not confs: return "0", "0.00", "0.00"
        return str(len(confs)), f"{np.mean(confs):.2f}", f"{np.max(confs):.2f}"

    count_std, avg_std, max_std = get_stats(conf_std)
    count_tile, avg_tile, max_tile = get_stats(conf_tile)
    count_hyb, avg_hyb, max_hyb = get_stats(conf_hyb)

    stats_data = [
        ["Metric", "Standard", "Tiled Only", "Hybrid (Final)"],
        ["Objects Detected", count_std, count_tile, count_hyb],
        ["Avg Confidence", avg_std, avg_tile, avg_hyb],
        ["Max Confidence", max_std, max_tile, max_hyb]
    ]

    # Print to Console
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Standard':<12} | {'Tiled Only':<12} | {'Hybrid':<12}")
    print("-" * 60)
    print(f"{'Objects Detected':<20} | {count_std:<12} | {count_tile:<12} | {count_hyb:<12}")
    print(f"{'Avg Confidence':<20} | {avg_std:<12} | {avg_tile:<12} | {avg_hyb:<12}")
    print(f"{'Max Confidence':<20} | {max_std:<12} | {max_tile:<12} | {max_hyb:<12}")
    print("="*60 + "\n")

    # 7. Visualization Grid
    print("Generating Dashboard...")
    fig = plt.figure(figsize=(16, 10))
    plt.suptitle("YOLO11 Low Light Detection Analysis", fontsize=16, fontweight='bold')
    
    # Grid: 2 rows of images, bottom area for table
    # Height ratios: Images get more space, table gets bottom 15%
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3])

    # -- Image 1: Original --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title("1. Original (Enhanced)", fontsize=12)
    ax1.axis('off')

    # -- Image 2: Standard --
    ax2 = fig.add_subplot(gs[0, 1])
    vis_std = draw_boxes(img_rgb, boxes_std, cls_std, conf_std, model.names, color=(255, 0, 0))
    ax2.imshow(vis_std)
    ax2.set_title(f"2. Standard Detection ({count_std} objects)", fontsize=12)
    ax2.axis('off')

    # -- Image 3: Tiled --
    ax3 = fig.add_subplot(gs[1, 0])
    vis_tile = draw_boxes(img_rgb, boxes_tile, cls_tile, conf_tile, model.names, color=(0, 0, 255))
    ax3.imshow(vis_tile)
    ax3.set_title(f"3. Tiled Detection ({count_tile} objects)", fontsize=12)
    ax3.axis('off')

    # -- Image 4: Hybrid --
    ax4 = fig.add_subplot(gs[1, 1])
    vis_hyb = draw_boxes(img_rgb, boxes_hyb, cls_hyb, conf_hyb, model.names, color=(0, 255, 0))
    ax4.imshow(vis_hyb)
    ax4.set_title(f"4. Hybrid Final ({count_hyb} objects)", fontsize=12)
    ax4.axis('off')

    # -- Table --
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    table = ax_table.table(
        cellText=stats_data[1:],
        colLabels=stats_data[0],
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2']*4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8) # Stretch height for readability

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Ensure bottom table has room
    plt.show()

if __name__ == "__main__":
    main()
