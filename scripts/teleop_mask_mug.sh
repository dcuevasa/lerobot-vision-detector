#!/bin/bash
set -e

# Fix serial permissions
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 2>/dev/null || true

# Source shared variables if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/set_variables.sh" ]; then
    . "$SCRIPT_DIR/set_variables.sh"
elif [ -f "set_variables.sh" ]; then
    . set_variables.sh
fi

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
LEADER_ID="${LEADER_ID:-leader_arm_test2}"

# Mask parameters:
# In COCO / YOLO segmentation models, mugs are classified under "cup".
# Specifying "cup,mug" matches either class.
TARGET_OBJECT="${1:-cup,mug}"
CAMERAS="${2:-all}"
MODEL_NAME="${3:-yolov8n-seg.pt}"

# Fallback dual-camera configuration if CAMERAS_JSON is unset
if [ -z "$CAMERAS_JSON" ]; then
CAMERAS_JSON='{
  "cam_high": {
    "type": "intelrealsense",
    "serial_number_or_name": "944622074682",
    "width": 640,
    "height": 480,
    "fps": 30,
    "use_depth": false
  },
  "cam_wrist": {
    "type": "opencv",
    "index_or_path": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0",
    "width": 640,
    "height": 480,
    "fps": 30
  }
}'
fi

echo "=================================================================="
echo "Starting Teleoperation with Instance Segmentation Mask over Mug"
echo "  Target:          $TARGET_OBJECT"
echo "  Mode:            seg (Instance Segmentation Mask)"
echo "  Model:           $MODEL_NAME"
echo "  Cameras:         $CAMERAS"
echo "  Follower Arm:    $FOLLOWER_ID ($FOLLOWER_PORT)"
echo "  Leader Arm:      $LEADER_ID ($LEADER_PORT)"
echo "=================================================================="

lerobot-vision-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS_JSON" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --detector.enabled=true \
    --detector.mode="seg" \
    --detector.model_name="$MODEL_NAME" \
    --detector.target_objects="$TARGET_OBJECT" \
    --detector.cameras="$CAMERAS" \
    "${@:4}"
