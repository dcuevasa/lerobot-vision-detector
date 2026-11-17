#!/bin/bash
set -e

TARGET_OBJECT="${1:-mug}"
SOURCE_COLOR="${2:-pink}"
DEST_COLOR="${3:-yellow}"
POLICY_PATH="${4:-bendca61/smolvla-so101-surface-transfer-pruned}"

TASK_DESC="Move the ${TARGET_OBJECT} from the ${SOURCE_COLOR} surface to the ${DEST_COLOR} surface"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

rm -rf "$HOME/.cache/huggingface/lerobot/local/eval_smolvla_unmasked_transfer"

CAMERAS_JSON='{"camera1": {"type": "intelrealsense", "serial_number_or_name": "944622074682", "width": 640, "height": 480, "fps": 30, "use_depth": false}, "camera2": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0", "width": 640, "height": 480, "fps": 30}}'

echo "=================================================================="
echo "Running Unmasked SmolVLA Surface Transfer Baseline (RTX 5070)"
echo "  Policy:          $POLICY_PATH"
echo "  Policy Prompt:   $TASK_DESC"
echo "  Detector:        DISABLED (raw camera1 + camera2)"
echo "=================================================================="

lerobot-vision-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --detector.enabled=false \
  --dataset.repo_id=local/eval_smolvla_unmasked_transfer \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${@:5}"
