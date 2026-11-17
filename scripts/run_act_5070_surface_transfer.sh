#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec ./run_masked_color_transfer_act_5070.sh "${1:-cup}" "${2:-pink}" "${3:-yellow}" "${4:-bendca61/act-so101-surface-transfer}" "${@:5}"
