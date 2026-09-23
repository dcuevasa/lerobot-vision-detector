# Standardized Models Directory

This directory stores pretrained weights, fine-tuned checkpoints, and open-vocabulary model files for `lerobot-vision-detector`.

When specifying `--model <name>` in any CLI command or script, `lerobot-vision-detector` automatically checks:
1. Exact file path (e.g. `/path/to/my_weights.pt`).
2. Inside this `models/` directory (e.g. `--model yolo11-detection-obj_s.pt` resolves to `models/yolo11-detection-obj_s.pt`).
3. Standard Ultralytics / Hugging Face model hub names (e.g. `yolov8n.pt`, `yolov8s-worldv2.pt`, `google/owlv2-base-patch16-ensemble`).

---

## Model Types Supported

### 1. Open-Vocabulary Models (`open_vocab` / `yolo_world` / `owlv2`)
Detect **any arbitrary object** described by natural language text on the fly without retraining:
- **`yolov8s-worldv2.pt`**: Fast real-time open-vocabulary detector. Specify any text prompt via `--target_objects` (e.g. `--target_objects "red mug, white tape, robot gripper"`).
- **`yolo11s-world.pt` / `yolo11m-world.pt`**: Latest YOLO-World architectures.
- **`google/owlv2-base-patch16-ensemble`**: Transformer-based open-vocabulary detector via Hugging Face Transformers.

### 2. Standard 2D Bounding Box Models (`bbox`)
Pretrained on COCO (80 classes, e.g. `cup`, `bottle`, `knife`, `scissors`):
- `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`

### 3. Instance Segmentation Models (`seg`)
Pretrained on COCO-Seg (80 classes with pixel-level masks):
- `yolov8n-seg.pt`, `yolo11n-seg.pt`

### 4. Custom Fine-Tuned Checkpoints
Store custom trained weights directly in this folder:
- e.g. `yolo11-detection-obj_s.pt`
