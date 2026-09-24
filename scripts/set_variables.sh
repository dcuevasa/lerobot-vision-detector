#!/bin/bash
rm -rf /home/sinfonia/.cache/huggingface/lerobot/local/so101_detected_dataset

# =====================================================================
# 1. HARDWARE & PORT SETUP
# =====================================================================
FOLLOWER_PORT="/dev/ttyACM0"
LEADER_PORT="/dev/ttyACM1"

FOLLOWER_ID="follower_arm_test4"
LEADER_ID="leader_arm_test3"

# Fix permissions automatically if serial devices exist
if [ -e "$FOLLOWER_PORT" ] || [ -e "$LEADER_PORT" ]; then
    sudo chmod 666 $FOLLOWER_PORT $LEADER_PORT 2>/dev/null || true
fi

# =====================================================================
# 2. FEATURE TOGGLES
# =====================================================================
ENABLE_EE_POSE="true"   # true/false: computes FK and injects ee_pos/quat
ENABLE_DEPTH="false"    # true/false: captures RealSense depth stream

# Camera selection toggles
USE_CAM_HIGH="true"     # true/false: Intel RealSense D435i
USE_CAM_WRIST="true"    # true/false: Arducam OV9281 wrist camera

# =====================================================================
# 3. VISION DETECTOR SETTINGS
# =====================================================================
DETECTOR_ENABLED="true"
DETECTOR_MODE="bbox"             # "bbox", "seg", or "both"
DETECTOR_TARGETS="cup"           # Object filter: e.g. "cup", "tape", "bottle", or "all"
DETECTOR_CAMERAS="cam_high"      # Comma-separated cameras to process, e.g. "cam_high" or "cam_high,cam_wrist"
DETECTOR_MODEL="yolov8n.pt"      # "yolov8n.pt" for bbox or "yolov8n-seg.pt" for seg
DETECTOR_CONF="0.25"

# =====================================================================
# 4. DATASET SETTINGS
# =====================================================================
REPO_ID="local/so101_detected_dataset"
NUM_EPISODES=1
TASK_DESC="pick the cup"

# =====================================================================
# 5. DYNAMIC CONFIGURATION BUILDER FOR CAMERAS
# =====================================================================
CAMERAS_JSON="{"
COMMA=""

if [ "$USE_CAM_HIGH" = "true" ]; then
    CAMERAS_JSON+='"cam_high": {"type": "intelrealsense", "serial_number_or_name": "944622074682", "width": 640, "height": 480, "fps": 30, "use_depth": '"$ENABLE_DEPTH"'}'
    COMMA=","
fi

if [ "$USE_CAM_WRIST" = "true" ]; then
    CAMERAS_JSON+="$COMMA \"cam_wrist\": {\"type\": \"opencv\", \"index_or_path\": \"/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0\", \"width\": 640, \"height\": 480, \"fps\": 30}"
fi
CAMERAS_JSON+="}"
