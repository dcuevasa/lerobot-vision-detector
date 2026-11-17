#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec ./eval_masked_color_transfer_smolvla.sh "${1:-cup}" "${2:-pink}" "${3:-yellow}" "${4:-bendca61/smolvla-so101-surface-transfer-gsam-obscure-init5}" "${@:5}"
