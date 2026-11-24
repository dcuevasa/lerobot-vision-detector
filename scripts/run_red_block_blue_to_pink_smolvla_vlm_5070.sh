#!/bin/bash
set -e

POLICY_PATH="${1:-bendca61/smolvla-so101-red-block-blue-to-pink-vlm}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
TASK_DESC="Move the red block from the blue surface to the pink surface"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

rm -rf "$HOME/.cache/huggingface/lerobot/local/eval_smolvla_vlm_red_block_blue_to_pink"

WRIST_CAM_UC599="/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0"
WRIST_CAM_UC684="/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_1080P_Low_Light_UC684-video-index0"
if [ -n "${WRIST_CAM_PATH:-}" ]; then
    WRIST_CAM="$WRIST_CAM_PATH"
elif [ -e "$WRIST_CAM_UC599" ]; then
    WRIST_CAM="$WRIST_CAM_UC599"
else
    WRIST_CAM="$WRIST_CAM_UC684"
fi

CAMERAS_JSON="{\"camera1\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"944622074682\", \"width\": 640, \"height\": 480, \"fps\": 30, \"use_depth\": false}, \"camera2\": {\"type\": \"opencv\", \"index_or_path\": \"${WRIST_CAM}\", \"width\": 640, \"height\": 480, \"fps\": 30}}"

echo "=================================================================="
echo "Running Unfrozen-VLM SmolVLA Specialist: Red Block (RTX 5070)"
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
  --dataset.repo_id=local/eval_smolvla_vlm_red_block_blue_to_pink \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${@:2}"
