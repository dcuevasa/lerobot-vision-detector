#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC_COLOR="${2:-pink}"
DST_COLOR="${3:-yellow}"
exec ./eval_masked_color_transfer_act.sh "${1:-cup}" "$SRC_COLOR" "$DST_COLOR" "${4:-bendca61/act-so101-init5-${SRC_COLOR}-to-${DST_COLOR}}" "${@:5}"
