#!/bin/bash
set -e

POLICY_PATH="${1:-bendca61/act-so101-red-block-blue-to-pink}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"
TASK_DESC="Move the red block from the blue surface to the pink surface"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

rm -rf "$HOME/.cache/huggingface/lerobot/local/eval_act_red_block_blue_to_pink"

CAMERAS_JSON='{"cam_high": {"type": "intelrealsense", "serial_number_or_name": "944622074682", "width": 640, "height": 480, "fps": 30, "use_depth": false}, "cam_wrist": {"type": "opencv", "index_or_path": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC599-video-index0", "width": 640, "height": 480, "fps": 30}}'

echo "=================================================================="
echo "Running Unmasked ACT Policy: Red Block Blue -> Pink (RTX 5070)"
echo "  Policy:          $POLICY_PATH"
echo "  Task Label:      $TASK_DESC"
echo "  Detector:        DISABLED (raw cam_high + cam_wrist)"
echo "=================================================================="

lerobot-vision-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --detector.enabled=false \
  --dataset.repo_id=local/eval_act_red_block_blue_to_pink \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=act \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${@:2}"
