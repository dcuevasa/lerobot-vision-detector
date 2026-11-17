#!/bin/bash
set -e

TARGET_OBJECT="${1:-cup}"
SOURCE_COLOR="${2:-pink}"
DEST_COLOR="${3:-yellow}"
POLICY_PATH="${4:-bendca61/xvla-so101-surface-transfer-gsam-obscure-init5}"

TASK_DESC="Move the masked object from the ${SOURCE_COLOR} surface to the ${DEST_COLOR} surface"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
DETECTOR_TYPE="${DETECTOR_TYPE:-grounded_sam}"
SAM_VERSION="${SAM_VERSION:-vit_b}"
INITIAL_FRAMES="${INITIAL_FRAMES:-5}"
DETECTOR_MODEL="${DETECTOR_MODEL:-$SAM_VERSION}"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

rm -rf "$HOME/.cache/huggingface/lerobot/local/eval_xvla_masked_transfer"

CAMERAS_JSON='{"image": {"type": "intelrealsense", "serial_number_or_name": "944622074682", "width": 640, "height": 480, "fps": 30, "use_depth": false}, "image2": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0", "width": 640, "height": 480, "fps": 30}}'

echo "=================================================================="
echo "Running X-VLA Masked-Object Surface Transfer (RTX 5070)"
echo "  Policy:          $POLICY_PATH"
echo "  Policy Prompt:   $TASK_DESC"
echo "  Detector Target: $TARGET_OBJECT ($DETECTOR_TYPE / $DETECTOR_MODEL, mode=obscure, init=${INITIAL_FRAMES}f)"
echo "=================================================================="

lerobot-vision-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --detector.enabled=true \
  --detector.detector_type="$DETECTOR_TYPE" \
  --detector.sam_version="$SAM_VERSION" \
  --detector.cameras="image" \
  --detector.mode=obscure \
  --detector.initial_frames="$INITIAL_FRAMES" \
  --detector.reuse_initial_detection=true \
  --detector.target_objects="$TARGET_OBJECT" \
  --detector.distractor_objects="robot" \
  --detector.model_name="$DETECTOR_MODEL" \
  --detector.conf_threshold=0.25 \
  --dataset.repo_id=local/eval_xvla_masked_transfer \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=xvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${@:5}"
