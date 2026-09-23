#!/bin/bash
set -e

# Directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Source Hugging Face token if available
if [ -f "$SCRIPT_DIR/../../scripts/coffee/dcuevas_hf_token.sh" ]; then
    . "$SCRIPT_DIR/../../scripts/coffee/dcuevas_hf_token.sh"
elif [ -f "dcuevas_hf_token.sh" ]; then
    . "dcuevas_hf_token.sh"
fi

INPUT_REPO="${1:-bendca61/so101-test-leader-v1}"
OUTPUT_REPO="${2:-${INPUT_REPO}-detected}"
CAMERAS="${3:-cam_high}"
MODE="${4:-bbox}"
TARGET_OBJECTS="${5:-cup}"
OUTPUT_DIR="./outputs/$(echo "$OUTPUT_REPO" | tr '/' '_')"

echo "=================================================================="
echo "Augmenting LeRobot Dataset with YOLO Spatial Detections"
echo "  Input Repo:      $INPUT_REPO"
echo "  Output Repo:     $OUTPUT_REPO"
echo "  Target Cameras:  $CAMERAS (unspecified cameras will NOT be processed)"
echo "  Mode:            $MODE (bbox or seg)"
echo "  Target Objects:  $TARGET_OBJECTS"
echo "  Output Dir:      $OUTPUT_DIR"
echo "=================================================================="

# Run offline dataset augmentation CLI
python -m lerobot_vision_detector.cli.augment \
  --input_repo "$INPUT_REPO" \
  --output_repo "$OUTPUT_REPO" \
  --cameras $CAMERAS \
  --mode "$MODE" \
  --target_objects "$TARGET_OBJECTS" \
  --output_dir "$OUTPUT_DIR" \
  --overwrite \
  "${@:6}"

echo "=================================================================="
echo "Augmentation complete! Dataset available at: $OUTPUT_DIR"
echo "=================================================================="
