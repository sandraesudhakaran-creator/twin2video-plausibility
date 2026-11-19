#!/usr/bin/env bash
# Run Cosmos-Transfer2.5 on the exported scene, then check the outputs.
# Needs a CUDA GPU machine with Cosmos-Transfer2.5 installed
# (https://github.com/nvidia-cosmos/cosmos-transfer2.5) — not a laptop.
#
#   1. on any machine:   twinplausi twin --out out/scene --cosmos-root /data/scene
#   2. copy out/scene to /data/scene on the GPU machine
#   3. on the GPU machine, from the cosmos-transfer2.5 checkout:
#        bash /path/to/run_cosmos.sh /data/scene
#   4. copy /data/scene/generated back and evaluate every output:
#        for v in out/scene/generated/*/*.mp4; do twinplausi evaluate --scene out/scene --video "$v"; done
set -euo pipefail
SCENE=${1:?usage: run_cosmos.sh /abs/path/to/scene}
for spec in "$SCENE"/specs/*.json; do
  echo "== $spec"
  python examples/inference.py --params_file "$spec"
done
