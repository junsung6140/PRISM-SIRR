#!/bin/bash
# PRISM training: LCC (Swap-Compose-Cycle) + LCS (InfoNCE) on FLUX.2 Klein 4B
#
# Usage:
#   bash scripts/train.sh
# Or override the config:
#   bash scripts/train.sh configs/your_config.yaml

set -e
CONFIG="${1:-configs/flux_klein_rr_cycle.yaml}"

accelerate launch train_flux_klein_rr_cycle.py --config "${CONFIG}"

echo "=========================================="
echo "PRISM training complete!"
echo "=========================================="
