#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. ./set_variables.sh

POLICY_PATH="${1:-local/my_trained_act_policy}"
TARGET_OBJECTS="${2:-cup}"
DETECTOR_MODE="${3:-bbox}"

echo "=================================================================="
echo "Evaluating Policy with Live Multi-Camera YOLO Detections"
echo "  Policy Path:     $POLICY_PATH"
echo "  Robot:           so101_physical_wrapped"
echo "  Detector Mode:   $DETECTOR_MODE"
echo "  Target Objects:  $TARGET_OBJECTS"
echo "  Wrapped Cameras: $DETECTOR_CAMERAS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.eval_policy \
    --robot.type=so101_physical_wrapped \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS_JSON" \
    --policy.path="$POLICY_PATH" \
    --detector.enabled=true \
    --detector.mode="$DETECTOR_MODE" \
    --detector.target_objects="$TARGET_OBJECTS" \
    --detector.cameras="$DETECTOR_CAMERAS" \
    --detector.model_name="$DETECTOR_MODEL" \
    --detector.conf_threshold=$DETECTOR_CONF \
    --display_data=true \
    "${@:4}"
