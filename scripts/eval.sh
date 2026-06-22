#!/bin/bash
# PRISM evaluation on SIRS benchmarks (real20, Nature, SIR2/{Postcard,Solid,Wild})
#
# Usage:
#   bash scripts/eval.sh
#   bash scripts/eval.sh configs/eval_flux_klein_rr.yaml

set -e
CONFIG="${1:-configs/eval_flux_klein_rr.yaml}"

python eval_flux_klein_rr.py --config "${CONFIG}"
