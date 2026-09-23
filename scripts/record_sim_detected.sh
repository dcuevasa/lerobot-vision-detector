#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

. ./set_variables.sh

REPO_ID="${1:-local/so101_mujoco_detected_demo}"
NUM_EPISODES="${2:-5}"
TASK_DESC="${3:-Put the block on the tray}"
DETECTOR_MODE="${4:-bbox}"
TARGET_OBJECTS="${5:-cube}"

echo "=================================================================="
echo "Recording MuJoCo SO-101 Dataset with Real-time YOLO Detections"
echo "  Robot:           so101_mujoco"
echo "  Teleop:          so101_ik"
echo "  Dataset Repo:    $REPO_ID"
echo "  Task:            $TASK_DESC"
echo "  Detector Mode:   $DETECTOR_MODE"
echo "  Target Objects:  $TARGET_OBJECTS"
echo "=================================================================="

python -m lerobot_vision_detector.cli.record \
  --robot.type=so101_mujoco \
  --robot.randomize_scene=true \
  --robot.reset_every_episode=true \
  --robot.camera_pos_base='[0.8, 0.2, 0.6]' \
  --robot.camera_euler_base='[0.0, -2.35619, -1.5708]' \
  --teleop.type=so101_ik \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK_DESC" \
  --dataset.episode_time_s=20 \
  --dataset.num_episodes=$NUM_EPISODES \
  --display_data=true \
  --detector.enabled=true \
  --detector.mode="$DETECTOR_MODE" \
  --detector.target_objects="$TARGET_OBJECTS" \
  --detector.cameras="realsense" \
  --detector.model_name="yolov8n.pt" \
  "${@:6}"
