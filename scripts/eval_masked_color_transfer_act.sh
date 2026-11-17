#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. ./set_variables.sh

TARGET_OBJECT="${1:-cup}"
SOURCE_COLOR="${2:-pink}"
DEST_COLOR="${3:-yellow}"
POLICY_PATH="${4:-bendca61/act-so101-surface-transfer}"

TASK_DESC="Move the masked object from the ${SOURCE_COLOR} surface to the ${DEST_COLOR} surface"
DETECTOR_SEG_MODEL="${DETECTOR_SEG_MODEL:-yolo11x-seg.pt}"
DETECTOR_TYPE="${DETECTOR_TYPE:-yolo}"

echo "=================================================================="
echo "Evaluating ACT: Masked-Object Color Surface Transfer (6-DoF)"
echo "  Policy Path:     $POLICY_PATH"
echo "  Task Label:      $TASK_DESC (ACT ignores text prompt)"
echo "  Detector Target: $TARGET_OBJECT ($DETECTOR_TYPE / $DETECTOR_SEG_MODEL, mode=obscure)"
echo "  Robot:           so101_follower ($FOLLOWER_ID on $FOLLOWER_PORT)"
echo "  Wrapped Cameras: $DETECTOR_CAMERAS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.eval_policy \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="$CAMERAS_JSON" \
    --policy.type=act \
    --policy.pretrained_path="$POLICY_PATH" \
    --policy.device=cuda \
    --single_task="$TASK_DESC" \
    --detector.enabled=true \
    --detector.detector_type="$DETECTOR_TYPE" \
    --detector.mode=obscure \
    --detector.target_objects="$TARGET_OBJECT" \
    --detector.distractor_objects="robot" \
    --detector.cameras="$DETECTOR_CAMERAS" \
    --detector.model_name="$DETECTOR_SEG_MODEL" \
    --detector.conf_threshold="$DETECTOR_CONF" \
    --display_data=true \
    "${@:5}"
