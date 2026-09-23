#!/bin/bash
set -e

sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/set_variables.sh" ]; then
    . "$SCRIPT_DIR/set_variables.sh"
fi

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
LEADER_ID="${LEADER_ID:-leader_arm_test2}"
TARGET_OBJECT="${1:-mug}"
DETECTOR_MODE="${2:-bbox}"

# Dual-camera configuration: cam_high (Intel RealSense) + cam_wrist (Arducam)
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

echo "=================================================================="
echo "Starting Teleoperation with Detections on BOTH Cameras"
echo "  Cameras:         cam_high (RealSense) & cam_wrist (Arducam)"
echo "  Target Object:   $TARGET_OBJECT"
echo "  Detector Mode:   $DETECTOR_MODE"
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
    --detector.mode="$DETECTOR_MODE" \
    --detector.target_objects="$TARGET_OBJECT" \
    --detector.cameras="cam_high,cam_wrist" \
    "${@:3}"
