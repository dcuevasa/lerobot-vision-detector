#!/bin/bash
set -e

SOURCE_COLOR="${1:-blue}"
DEST_COLOR="${2:-pink}"
TARGET_OBJECT="${3:-red block}"
POLICY_PATH="${4:-bendca61/smolvla-so101-surface-transfer-and-red-block-mix}"
EXTRA_ARGS=("${@:5}")

TASK_DESC="Move the ${TARGET_OBJECT} from the ${SOURCE_COLOR} surface to the ${DEST_COLOR} surface"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm_test4}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/eval_smolvla_red_block_blue_to_pink_${RUN_TS}}"
EVAL_LOG_DIR="${LEROBOT_EVAL_LOG_DIR:-$PWD/eval_logs}"

if [ -e "$FOLLOWER_PORT" ]; then
    sudo chmod 666 "$FOLLOWER_PORT" 2>/dev/null || true
fi

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

AUTO_POLICY_ARGS=()
while IFS= read -r _arg; do
    [[ "$_arg" == --policy.* ]] && AUTO_POLICY_ARGS+=("$_arg")
done < <(python3 -m lerobot_vision_detector.cli.rollout_logging --print-policy-args "$POLICY_PATH" --policy-type smolvla -- "${EXTRA_ARGS[@]}" 2>/dev/null || true)

analyze_on_exit() {
    local _ds_dir="$HOME/.cache/huggingface/lerobot/${DATASET_REPO_ID}"
    if [ -d "${_ds_dir}/data" ]; then
        echo "Analyzing & logging rollout telemetry for ${DATASET_REPO_ID}..."
        python3 -m lerobot_vision_detector.cli.rollout_logging \
          --dataset-root "$_ds_dir" \
          --dataset-repo-id "$DATASET_REPO_ID" \
          --policy-path "$POLICY_PATH" \
          --policy-type smolvla \
          --task "$TASK_DESC" \
          --log-dir "$EVAL_LOG_DIR" \
          --extra-args "${EXTRA_ARGS[*]:-}" \
          --outcome "${EVAL_OUTCOME:-recorded}" \
          --notes "${EVAL_NOTES:-}" || true
    fi
}
trap analyze_on_exit EXIT

echo "=================================================================="
echo "Running Unmasked Mixed SmolVLA Policy (RTX 5070)"
echo "  Policy:          $POLICY_PATH"
echo "  Synced Args:     ${AUTO_POLICY_ARGS[*]:-none}"
echo "  Dataset Repo:    $DATASET_REPO_ID"
echo "  Eval Log Dir:    $EVAL_LOG_DIR"
echo "  Policy Prompt:   $TASK_DESC"
echo "  Detector:        DISABLED (raw camera1 + camera2)"
echo "=================================================================="

lerobot-vision-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --detector.enabled=false \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.device=cuda \
  --display_data=true \
  "${AUTO_POLICY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
