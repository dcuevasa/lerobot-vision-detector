#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. ./set_variables.sh

POLICY_PATH="${1:-bendca61/xvla-so101-surface-transfer-gsam-obscure}"
TASK_DESC="${2:-Move the mug from the pink surface to the yellow surface}"
TARGET_OBJECTS="${3:-cup}"
DETECTOR_MODE="${4:-obscure}"
DETECTOR_SEG_MODEL="${DETECTOR_SEG_MODEL:-yolo11x-seg.pt}"
DETECTOR_TYPE="${DETECTOR_TYPE:-yolo}"

echo "=================================================================="
echo "Evaluating X-VLA Policy with Live Multi-Camera Detections (6-DoF)"
echo "  Policy Path:     $POLICY_PATH"
echo "  Task Prompt:     $TASK_DESC"
echo "  Robot:           so101_follower ($FOLLOWER_ID on $FOLLOWER_PORT)"
echo "  Detector Type:   $DETECTOR_TYPE ($DETECTOR_SEG_MODEL)"
echo "  Detector Mode:   $DETECTOR_MODE"
echo "  Target Objects:  $TARGET_OBJECTS"
echo "  Wrapped Cameras: $DETECTOR_CAMERAS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.eval_policy \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="$CAMERAS_JSON" \
    --policy.type=xvla \
    --policy.pretrained_path="$POLICY_PATH" \
    --policy.device=cuda \
    --single_task="$TASK_DESC" \
    --rename_map='{"observation.images.cam_high": "observation.images.image", "observation.images.cam_wrist": "observation.images.image2"}' \
    --detector.enabled=true \
    --detector.detector_type="$DETECTOR_TYPE" \
    --detector.mode="$DETECTOR_MODE" \
    --detector.target_objects="$TARGET_OBJECTS" \
    --detector.distractor_objects="robot" \
    --detector.cameras="$DETECTOR_CAMERAS" \
    --detector.model_name="$DETECTOR_SEG_MODEL" \
    --detector.conf_threshold="$DETECTOR_CONF" \
    --display_data=true \
    "${@:5}"
