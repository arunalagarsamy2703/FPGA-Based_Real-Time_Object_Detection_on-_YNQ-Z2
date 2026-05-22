# FPGA-Based Real-Time Object Detection on PYNQ-Z2

Real-time object detection on the PYNQ-Z2 FPGA platform using YOLOv3-tiny, AXI DMA communication, OpenCV, and hardware-software co-design for embedded AI applications.

---

# Project Overview

This project implements a real-time object detection system on the PYNQ-Z2 FPGA board using YOLOv3-tiny and AXI DMA communication. The implementation combines FPGA hardware acceleration with software-based deep learning inference to demonstrate embedded AI on a resource-constrained platform.

The system uses:
- Processing System (PS) for image acquisition and object detection
- Programmable Logic (PL) for AXI DMA communication
- YOLOv3-tiny for lightweight real-time inference
- OpenCV for preprocessing and visualization

The implementation demonstrates successful PS-PL communication using AXI DMA and AXI4-Stream FIFO loopback architecture. :contentReference[oaicite:0]{index=0}

---

# Features

- Real-time object detection using YOLOv3-tiny
- FPGA-based embedded AI implementation
- AXI DMA communication between PS and PL
- Hardware-software co-design architecture
- CLAHE image enhancement preprocessing
- FPGA overlay loading using PYNQ
- DMA transfer optimization
- Real-time frame processing
- Label remapping for improved readability
- Vivado IP Integrator hardware design

---

# Hardware Used

| Component | Description |
|---|---|
| FPGA Board | PYNQ-Z2 |
| FPGA Device | Xilinx Zynq-7020 |
| Processor | ARM Cortex-A9 |
| Camera | USB Webcam |
| Storage | Micro SD Card |

---

# Software and Tools

| Tool | Purpose |
|---|---|
| Vivado 2023.1 | FPGA Hardware Design |
| PYNQ v2.7 | FPGA Software Framework |
| Python 3.8 | Application Development |
| OpenCV | Image Processing |
| Jupyter Notebook | Runtime Environment |
| YOLOv3-tiny | Object Detection |
| NumPy | Numerical Operations |

---

# System Architecture

```text
USB Camera
    ↓
CLAHE Preprocessing
    ↓
AXI DMA Transfer
    ↓
AXI4-Stream FIFO Loopback
    ↓
YOLOv3-tiny Detection
    ↓
Bounding Box Output
```

---

# Hardware Design

## Vivado Block Design

The hardware architecture contains:
- ZYNQ7 Processing System
- AXI DMA
- AXI4-Stream FIFO
- AXI Interconnect
- Processor System Reset

### Hardware Files

```text
hardware/
├── design_1_wrapper.bit
├── design_1_wrapper.hwh
└── design_1.pdf
```

---

# Project Structure

```text
FPGA-Based-Real-Time-Object-Detection-on-PYNQ-Z2/
│
├── hardware/
│   ├── design_1_wrapper.bit
│   ├── design_1_wrapper.hwh
│   └── design_1.pdf
│
├── software/
│   ├── object_detection.py
│   ├── dma_transfer.py
│   ├── clahe_preprocessing.py
│   └── overlay_loader.py
│
├── notebook/
│   └── Real_Time_Object_Detection.ipynb
│
├── model/
│   ├── yolov3-tiny.cfg
│   ├── yolov3-tiny.weights
│   └── coco.names
│
├── output/
│   └── latest_detection.jpg
│
├── docs/
├── images/
├── README.md
├── requirements.txt
└── LICENSE
```

---

# How to Run

## Step 1: Copy Hardware Files to PYNQ Board

Copy:
```text
design_1_wrapper.bit
design_1_wrapper.hwh
```

to the PYNQ board.

---

## Step 2: Copy YOLO Model Files

Copy:
```text
yolov3-tiny.cfg
yolov3-tiny.weights
coco.names
```

to the `model/` folder.

---

## Step 3: Open Jupyter Notebook

Open the PYNQ Jupyter environment:

```text
http://<PYNQ_IP>:9090
```

Example:
```text
http://192.168.1.10:9090
```

---

## Step 4: Run Object Detection

```bash
python3 object_detection.py
```

or run the notebook:
```text
Real_Time_Object_Detection.ipynb
```

---

## Step 5: View Detection Results

Open:
```text
latest_detection.jpg
```

to view object detection results.

---

# Overlay Loading Example

```python
from pynq import Overlay

overlay = Overlay("design_1_wrapper.bit")
dma = overlay.axi_dma_0
```

---

# DMA Optimization

Initially, full image frames were transferred using DMA communication:

| Parameter | Original | Optimized |
|---|---|---|
| Resolution | 640×480×3 | 64×64×1 |
| Data Size | 921600 bytes | 4096 bytes |

The optimized grayscale thumbnail significantly improved:
- DMA communication efficiency
- PS-PL interaction
- Memory utilization
- Transfer speed

---

# CLAHE Preprocessing

CLAHE (Contrast Limited Adaptive Histogram Equalization) improves image visibility under:
- Low-light conditions
- Uneven illumination
- Poor contrast environments

### CLAHE Parameters

| Parameter | Value |
|---|---|
| Clip Limit | 2.0 |
| Tile Grid Size | 8×8 |

Example:

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(gray_image)
```

---

# YOLOv3-tiny Detection Parameters

| Parameter | Value |
|---|---|
| Confidence Threshold | 0.15 |
| NMS Threshold | 0.30 |
| Dataset | COCO |
| Object Classes | 80 |

---

# Results

The implementation successfully detected:
- person
- laptop
- bottle
- chair
- cup
- keyboard
- mouse

### Performance

| Metric | Value |
|---|---|
| FPS Range | 1–3 FPS |
| Average FPS | ~2 FPS |
| Maximum Objects Detected | 8 |

The implementation successfully demonstrated:
- Real-time object detection
- Stable PS-PL communication
- Embedded AI acceleration
- FPGA-based hardware interaction

Real-time object detection on PYNQ-Z2 using YOLO architectures has been explored in several FPGA research and open-source projects. :contentReference[oaicite:1]{index=1}

---

# Detection Pipeline

```text
Live Camera Input
        ↓
CLAHE Enhancement
        ↓
DMA Communication
        ↓
FIFO Loopback Verification
        ↓
YOLOv3-tiny Inference
        ↓
NMS Filtering
        ↓
Label Remapping
        ↓
Final Detection Output
```

---

# Download Model Weights

`yolov3-tiny.weights` is too large for GitHub.

Download it from:

:contentReference[oaicite:2]{index=2}

Place the file inside:

```text
model/
```

folder on the PYNQ board.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Bathreesh/FPGA-Based-Real-Time-Object-Detection-on-PYNQ-Z2.git
```

## Move to Repository

```bash
cd FPGA-Based-Real-Time-Object-Detection-on-PYNQ-Z2
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Applications

- Smart surveillance systems
- Embedded AI applications
- Robotics
- Traffic monitoring
- Industrial automation
- Edge AI systems

---

# Future Improvements

- Full FPGA acceleration for CNN inference
- YOLOv5 / YOLOv8 integration
- FPGA-based image preprocessing acceleration
- HDMI real-time streaming
- IoT and cloud integration
- Multi-channel DMA optimization

---

# References

1. YOLOv3: An Incremental Improvement  
2. Xilinx Zynq-7000 Documentation  
3. AXI DMA Product Guide  
4. OpenCV Library  
5. PYNQ Documentation  

---

# Author

Bathreesh

