# lerobot-vision-detector

**`lerobot-vision-detector`** is an extensible computer vision pipeline designed for the **LeRobot** ecosystem. It bridges deep vision models (YOLO bounding boxes and instance segmentation masks) with robotic manipulation datasets and live camera streams.

It is purpose-built to solve two primary robotics workflows:
1. **Offline Dataset Augmentation:** Ingest existing LeRobot datasets, run object detection or segmentation for specific target objects on **user-specified cameras** (leaving unspecified cameras untouched), and save/publish the augmented dataset to Hugging Face Hub under a new name.
2. **Real-Time Multi-Camera Wrapping:** Wrap multiple robot cameras at once with live YOLO bounding box or instance segmentation rendering for **both training** (recording detected episodes directly) and **running downstream policies** (such as ACT, Diffusion Policy, SmolVLM) in real time.

---

## Architecture Overview

```
                                 +------------------------------+
                                 |  YOLO / Vision Model Backend |
                                 | (YOLO-Bbox, YOLO-Seg, etc.)  |
                                 +--------------+---------------+
                                                |
                     +--------------------------+--------------------------+
                     |                                                     |
                     v                                                     v
      +------------------------------+                      +------------------------------+
      |  Offline Dataset Augmentor   |                      |  Real-Time Camera Wrappers   |
      |   (DatasetVideoAugmentor)    |                      | (MultiCameraVisionWrapper)   |
      +--------------+---------------+                      +--------------+---------------+
                     |                                                     |
     Processes ONLY Specified Cameras                      Wraps ONLY Specified Cameras
     Unspecified Cameras are Preserved                     Unspecified Cameras are Untouched
                     |                                                     |
                     v                                                     v
      +------------------------------+                      +------------------------------+
      |  New LeRobotDataset + Videos |                      | Live Observations / Training |
      |  + lerobot_annotations.json  |                      | & Evaluation (ACT, Diffusion)|
      |  + Publish to Hugging Face   |                      | + Live Rerun Stream          |
      +------------------------------+                      +------------------------------+
```

---

## 1. Quickstart & Installation

This project is configured to use the same pyenv (`lerobot`) as the parent workspace:

```bash
cd lerobot-vision-detector
pip install -e . --no-deps
```

> **Note on YOLO (Ultralytics):**
> `lerobot-vision-detector` automatically discovers `ultralytics` if it is installed in the active environment or in sister pyenv environments (such as `yolo`). Alternatively, install it directly:
> ```bash
> pip install ultralytics
> ```

---

## 2. Core Feature 1: Offline Dataset Augmentation & Publishing

Take any existing recorded LeRobot dataset (from local disk or downloaded from Hugging Face Hub), detect the target object(s) across **specified cameras**, render either bounding boxes or instance masks onto every frame of the video, and publish the new dataset to Hugging Face Hub.

### Rule on Camera Processing:
- **Specified cameras:** Every frame of every video is processed through the vision detector and rendered with the bounding box / segmentation mask.
- **Unspecified cameras:** **DO NOT PROCESS.** Unspecified cameras remain completely unmodified and preserved in their original pixel format.

### Command-Line Usage:

```bash
# Augment 'cam_high' with bounding boxes for "cup" and publish locally:
python -m lerobot_vision_detector.cli.augment \
    --input_repo "bendca61/so101-test-leader-v1" \
    --output_repo "bendca61/so101-test-leader-v1-cup-bbox" \
    --cameras cam_high \
    --mode bbox \
    --target_objects "cup" \
    --overwrite

# Augment 'cam_high' and 'cam_wrist' with segmentation masks for "cup" and "tape" and push to Hugging Face Hub:
python -m lerobot_vision_detector.cli.augment \
    --input_repo "bendca61/so101-test-leader-v1" \
    --output_repo "bendca61/so101-test-leader-v1-segmented" \
    --cameras cam_high cam_wrist \
    --mode seg \
    --target_objects "cup,tape" \
    --push_to_hub \
    --overwrite
```

### Ready-to-Use Bash Script:

Inspired by `scripts/coffee/split_subtasks.sh`:

```bash
bash scripts/augment_dataset.sh \
    "bendca61/so101-test-leader-v1" \
    "bendca61/so101-test-leader-v1-cup-bbox" \
    "cam_high" \
    "bbox" \
    "cup" \
    --push_to_hub
```

### Python API:

```python
from lerobot_vision_detector import DatasetVideoAugmentor, make_detector

# Initialize detector (Bbox or Seg)
detector = make_detector(
    detector_type="yolo",
    mode="seg",                 # "bbox", "seg", or "both"
    target_objects=["cup"],     # Single object or list of objects
    conf_threshold=0.25,
)

augmentor = DatasetVideoAugmentor(detector=detector)
out_path = augmentor.augment(
    input_repo="bendca61/so101-test-leader-v1",
    output_repo="bendca61/so101-test-leader-v1-cup-seg",
    cameras=["cam_high"],       # Unspecified cameras like 'cam_wrist' remain untouched!
    output_dir="./outputs/so101_cup_seg",
    push_to_hub=True,
    overwrite=True,
)
```

---

## 3. Core Feature 2: Real-Time Multi-Camera Wrapping

Wrap multiple cameras simultaneously on any LeRobot robot instance. This serves two core robotics use cases:

1. **Recording for Policy Training:**
   During teleoperation or recording sessions, wrapped cameras feed YOLO-annotated frames directly into the recorded video streams. Downstream policies (ACT, Diffusion) are then trained directly on these visual spatial cues without requiring any architecture changes!
2. **Running Policies in Real Time:**
   During real-time policy evaluation or closed-loop teleoperation, the cameras run YOLO in real time and feed the exact same visual detections into the policy network.

### In-Place Robot Wrapping:

```python
from lerobot.robots import make_robot_from_config
from lerobot_vision_detector import wrap_robot_cameras

# 1. Instantiate any LeRobot robot (SO-101 physical, MuJoCo, etc.)
robot = make_robot_from_config(robot_cfg)

# 2. Wrap specified cameras in-place
wrap_robot_cameras(
    robot=robot,
    camera_keys=["cam_high"],    # Only cam_high is processed; cam_wrist is untouched!
    mode="bbox",                 # "bbox" or "seg"
    target_objects="cup",        # Filter for "cup"
    model_name="yolov8n.pt",     # YOLO weights
    conf_threshold=0.25,
)

# 3. Use robot normally!
# robot.get_observation() automatically provides frames with detections for cam_high
obs = robot.get_observation()
```

### Multi-Camera Dictionary Wrapping:

```python
from lerobot_vision_detector import wrap_cameras

wrapped_cameras = wrap_cameras(
    cameras=robot.cameras,
    camera_keys=["cam_high", "realsense"],  # Unspecified cameras untouched
    mode="seg",
    target_objects=["cup", "bottle"],
)
```

---

## 4. Hardware & Teleoperation Scripts

Inspired by `FLAG-Embodied-data` and `scripts/coffee`:

| Script | Purpose |
| --- | --- |
| `scripts/list_classes.sh` | Lists and searches all available detectable object classes for any YOLO model. |
| `scripts/set_variables.sh` | Shared ports (`/dev/ttyACM*`), IDs, camera JSON toggles, and detector parameters. |
| `scripts/augment_dataset.sh` | Augments existing datasets offline with bbox or segmentation masks and uploads to HF Hub. |
| `scripts/record_physical_detected.sh` | Records SO-101 arm episodes with real-time YOLO camera wrapping & Rerun stream. |
| `scripts/record_sim_detected.sh` | Records MuJoCo simulation episodes with real-time YOLO camera wrapping. |
| `scripts/teleop_detected.sh` | Real-time teleoperation with leader arm and live YOLO detections rendered in Rerun. |
| `scripts/cameras_teleop_detected.sh` | Teleoperates on both physical cameras (`cam_high` + `cam_wrist`) with detections. |
| `scripts/eval_detected.sh` | Runs trained policies (ACT / Diffusion) with real-time camera detections. |

---

## 5. Discovering Available Model Classes

To see exactly what object names a model recognizes (e.g. `cup`, `bottle`, `knife`, `scissors`):

```bash
# Print all classes for yolov8n.pt (default):
lerobot-vision-list-classes

# Search for a specific keyword in a model:
lerobot-vision-list-classes --search cup

# Check classes for a custom checkpoint:
lerobot-vision-list-classes --model path/to/custom_weights.pt

# Or using the helper script:
bash scripts/list_classes.sh --search cup
```

### Manual Episode Recording Controls:
When recording episodes with `scripts/record_physical_detected.sh` or `lerobot-vision-record`:
- `Enter`: Start episode recording.
- `Right`: Accept and save episode.
- `Left`: Discard buffered episode.
- `Space`: Skip / retry attempt without saving.
- `Esc`: Stop recording session and finalize dataset.

---

## 5. Extensibility & Custom Detectors

The detector architecture is modular and decoupled from LeRobot hardware:

```python
from lerobot_vision_detector.detectors import BaseDetector, DetectionResult, register_detector

class CustomSAMDetector(BaseDetector):
    def predict(self, image: np.ndarray) -> DetectionResult:
        # Run custom SAM / Mask R-CNN / RT-DETR model...
        return DetectionResult(...)

register_detector("sam", CustomSAMDetector)
```

---

## 6. Running Tests

Run the full verification test suite (13 unit and integration tests):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```