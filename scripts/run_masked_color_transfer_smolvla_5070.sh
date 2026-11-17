#!/bin/bash
set -e

TARGET_OBJECT="${1:-cup}"
SOURCE_COLOR="${2:-pink}"
DEST_COLOR="${3:-yellow}"
POLICY_PATH="${4:-bendca61/smolvla-so101-surface-transfer-gsam-obscure}"

TASK_DESC="Move the masked object from the ${SOURCE_COLOR} surface to the ${DEST_COLOR} surface"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
DETECTOR_MODEL="${DETECTOR_MODEL:-yolo11x-seg.pt}"
DETECTOR_TYPE="${DETECTOR_TYPE:-yolo}"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

rm -rf "$HOME/.cache/huggingface/lerobot/local/eval_smolvla_masked_transfer"

CAMERAS_JSON='{"camera1": {"type": "intelrealsense", "serial_number_or_name": "944622074682", "width": 640, "height": 480, "fps": 30, "use_depth": false}, "camera2": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0", "width": 640, "height": 480, "fps": 30}}'

echo "=================================================================="
echo "Running SmolVLA Masked-Object Surface Transfer (RTX 5070)"
echo "  Policy:          $POLICY_PATH"
echo "  Policy Prompt:   $TASK_DESC"
echo "  Detector Target: $TARGET_OBJECT ($DETECTOR_TYPE / $DETECTOR_MODEL, mode=obscure)"
echo "=================================================================="

lerobot-vision-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --detector.enabled=true \
  --detector.detector_type="$DETECTOR_TYPE" \
  --detector.cameras="camera1" \
  --detector.mode=obscure \
  --detector.target_objects="$TARGET_OBJECT" \
  --detector.distractor_objects="robot" \
  --detector.model_name="$DETECTOR_MODEL" \
  --detector.conf_threshold=0.25 \
  --dataset.repo_id=local/eval_smolvla_masked_transfer \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${@:5}"
