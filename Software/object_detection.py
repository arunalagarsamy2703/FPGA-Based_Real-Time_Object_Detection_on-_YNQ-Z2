import cv2
import numpy as np
from pynq import Overlay, allocate

# =========================================================
# 1. LOAD FPGA OVERLAY
# =========================================================
print("Loading overlay...", flush=True)

ol  = Overlay("/home/xilinx/jupyter_notebooks/design_2_wrapper.bit")
dma = ol.axi_dma_0

print("Overlay loaded!", flush=True)

# =========================================================
# 2. LOAD YOLOv3-TINY
# =========================================================
print("Loading YOLO model...", flush=True)

net = cv2.dnn.readNetFromDarknet(
    "/home/xilinx/jupyter_notebooks/yolov3-tiny.cfg",
    "/home/xilinx/jupyter_notebooks/yolov3-tiny.weights"
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("YOLO model loaded!", flush=True)

layer_names   = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# =========================================================
# 3. COCO CLASSES
# =========================================================
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

LABEL_REMAP = {
    "vase": "bottle",
    "tv":   "laptop",
    "bowl": "cup",
}

# =========================================================
# 4. USB CAMERA SETUP
# =========================================================
print("Opening USB camera...", flush=True)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera 0 not found! Trying camera 1...", flush=True)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("No USB camera detected!")

print("USB Camera connected!", flush=True)

# =========================================================
# 5. IMAGE ENHANCEMENT
# =========================================================
clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

def enhance_frame(frame):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

# =========================================================
# 6. DMA TRANSFER
# =========================================================
DMA_W = 64
DMA_H = 64

dma_size = DMA_W * DMA_H

input_buffer  = allocate(shape=(dma_size,), dtype=np.uint8)
output_buffer = allocate(shape=(dma_size,), dtype=np.uint8)

def dma_transfer(frame):

    thumb = cv2.resize(frame, (DMA_W, DMA_H))

    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

    np.copyto(input_buffer, gray.flatten())

    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(output_buffer)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    return frame

# =========================================================
# 7. POSTPROCESS
# =========================================================
def postprocess(outputs, frame,
                conf_thresh=0.25,
                nms_thresh=0.3):

    h, w = frame.shape[:2]

    boxes       = []
    confidences = []
    class_ids   = []

    for output in outputs:

        for det in output:

            scores = det[5:]

            class_id = np.argmax(scores)

            confidence = float(scores[class_id])

            if confidence < conf_thresh:
                continue

            cx = int(det[0] * w)
            cy = int(det[1] * h)

            bw = int(det[2] * w)
            bh = int(det[3] * h)

            x = max(0, int(cx - bw / 2))
            y = max(0, int(cy - bh / 2))

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_thresh,
        nms_thresh
    )

    detected = []

    if len(indices) > 0:

        for i in indices.flatten():

            x, y, bw, bh = boxes[i]

            label_name = CLASSES[class_ids[i]]

            label_name = LABEL_REMAP.get(label_name, label_name)

            conf = confidences[i]

            label = f"{label_name}: {conf:.2f}"

            detected.append(label)

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cv2.rectangle(
                frame,
                (x, y - th - 10),
                (x + tw, y),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    return frame, detected

# =========================================================
# 8. DISPLAY RESULT IN NOTEBOOK
# =========================================================
def show_frame_in_notebook(frame,
                           frame_num,
                           total,
                           detected):

    from IPython.display import display, Image as IPImage

    cv2.putText(
        frame,
        f"Frame: {frame_num}/{total}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Objects: {len(detected)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    _, buf = cv2.imencode('.jpg', frame)

    display(IPImage(data=buf.tobytes()))

# =========================================================
# 9. MAIN LOOP
# =========================================================
MAX_FRAMES  = 30
frame_count = 0

print("=" * 50, flush=True)
print(f"Starting detection ({MAX_FRAMES} frames)", flush=True)
print("=" * 50, flush=True)

try:

    while frame_count < MAX_FRAMES:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame!", flush=True)
            continue

        # Enhance image
        enhanced = enhance_frame(frame)

        # DMA transfer
        processed = dma_transfer(enhanced)

        # YOLO inference
        blob = cv2.dnn.blobFromImage(
            processed,
            scalefactor=1/255.0,
            size=(416, 416),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)

        outputs = net.forward(output_layers)

        # Postprocess
        result, detected = postprocess(
            outputs,
            processed.copy()
        )

        # Save latest image
        cv2.imwrite(
            "/home/xilinx/jupyter_notebooks/latest_detection.jpg",
            result
        )

        # Show in notebook
        show_frame_in_notebook(
            result,
            frame_count + 1,
            MAX_FRAMES,
            detected
        )

        # Console output
        print(
            f"Frame {frame_count+1}/{MAX_FRAMES} — Objects: {len(detected)}",
            flush=True
        )

        for d in detected:
            print(f"  Detected: {d}", flush=True)

        print("-" * 50, flush=True)

        frame_count += 1

except KeyboardInterrupt:

    print("\nStopped by user", flush=True)

finally:

    print("\nCleaning up...", flush=True)

    cap.release()

    input_buffer.freebuffer()
    output_buffer.freebuffer()

    print(f"Done! Processed {frame_count} frames", flush=True)

    print("Saved: latest_detection.jpg", flush=True)