#!/usr/bin/env bash
#
# Run eotf, llamea, zero_shot, and gp_baseline on the PORTAL benchmark
# (fids 1..12, dim=2, iid=1) using google/gemini-2.0-flash-001 for the
# LLM-driven methods. Timing CSVs land under time_measurements/.
#
# The 12 active PORTAL instances (fids 1..12 in sorted order):
#   1  P37_H1_K12_Smooth
#   2  P44_I4_Additive_SubQuad
#   3  P109_InterferenceMesh
#   4  P111_RadialBraid
#   5  P114_BandedQuartet
#   6  P208_TightCluster
#   7  P217_WaveletStorm2
#   8  P222_FloweryBloom
#   9  P308_OctetMixed
#  10  P309_TripleNeedles
#  11  P315_TerracedRings
#  12  P318_CrossHatch
# Other generated instances are kept in portal/instances_archive/.
#
# Usage (from project root):
#   ./ela_guided_llm_bench/experiments/run_portal_benchmark.sh

set -euo pipefail

MODEL="google/gemini-2.0-flash-001"
DIM=2
IID=1
START_FID=1
END_FID=12

cd "$(dirname "$0")/../.."

for METHOD in eotf llamea zero_shot; do
    echo "=========================================="
    echo "  $METHOD  (model=$MODEL)"
    echo "=========================================="
    poetry run python ela_guided_llm_bench/experiments/main.py \
        --problem-class PORTAL \
        --start-fid "$START_FID" \
        --end-fid "$END_FID" \
        --method "$METHOD" \
        --model "$MODEL" \
        --dim "$DIM" \
        --iid "$IID"
done

echo "=========================================="
echo "  gp_baseline"
echo "=========================================="
poetry run python ela_guided_llm_bench/experiments/gp_baseline_main.py \
    --problem-class portal \
    --start-fid "$START_FID" \
    --end-fid "$END_FID" \
    --dim "$DIM" \
    --iid "$IID"

echo
echo "All PORTAL runs finished."
