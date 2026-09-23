#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load variables and camera configurations
. ./set_variables.sh

# Source HF token if available
if [ -f "$SCRIPT_DIR/../../scripts/coffee/dcuevas_hf_token.sh" ]; then
    . "$SCRIPT_DIR/../../scripts/coffee/dcuevas_hf_token.sh"
fi

echo "=================================================================="
echo "Recording Physical SO-101 Dataset with Real-time YOLO Detections"
echo "  Follower Arm:    $FOLLOWER_ID on $FOLLOWER_PORT"
echo "  Leader Arm:      $LEADER_ID on $LEADER_PORT"
echo "  Dataset Repo:    $REPO_ID"
echo "  Task:            $TASK_DESC"
echo "  Detector Mode:   $DETECTOR_MODE"
echo "  Target Objects:  $DETECTOR_TARGETS"
echo "  Wrapped Cameras: $DETECTOR_CAMERAS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.record \
    --robot.type=so101_physical_wrapped \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="$CAMERAS_JSON" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --dataset.repo_id="$REPO_ID" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=$NUM_EPISODES \
    --dataset.single_task="$TASK_DESC" \
    --detector.enabled=$DETECTOR_ENABLED \
    --detector.mode="$DETECTOR_MODE" \
    --detector.target_objects="$DETECTOR_TARGETS" \
    --detector.cameras="$DETECTOR_CAMERAS" \
    --detector.model_name="$DETECTOR_MODEL" \
    --detector.conf_threshold=$DETECTOR_CONF \
    "$@"
