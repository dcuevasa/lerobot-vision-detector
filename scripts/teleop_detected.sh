#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. ./set_variables.sh

echo "=================================================================="
echo "Starting Teleoperation with Live Multi-Camera YOLO Detections"
echo "  Follower:        $FOLLOWER_ID on $FOLLOWER_PORT"
echo "  Leader:          $LEADER_ID on $LEADER_PORT"
echo "  Detector Mode:   $DETECTOR_MODE"
echo "  Target Objects:  $DETECTOR_TARGETS"
echo "  Wrapped Cameras: $DETECTOR_CAMERAS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS_JSON" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --detector.enabled=$DETECTOR_ENABLED \
    --detector.mode="$DETECTOR_MODE" \
    --detector.target_objects="$DETECTOR_TARGETS" \
    --detector.cameras="$DETECTOR_CAMERAS" \
    --detector.model_name="$DETECTOR_MODEL" \
    --detector.conf_threshold=$DETECTOR_CONF \
    "$@"
