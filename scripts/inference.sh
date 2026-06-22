#!/bin/bash
# PRISM inference on arbitrary images (no GT required)
#
# Usage:
#   bash scripts/inference.sh
#   bash scripts/inference.sh configs/inference_flux_klein_rr.yaml

set -e
CONFIG="${1:-configs/inference_flux_klein_rr.yaml}"

python inference_flux_klein_rr.py --config "${CONFIG}"
