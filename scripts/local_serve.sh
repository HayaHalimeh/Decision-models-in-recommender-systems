#!/usr/bin/env bash
# Serves one local model on the endpoint its config uses (configs/models/llm.yaml, configs/models/decision.yaml):
#
#   bash scripts/local_serve.sh qwen35_9b            # vLLM, http://localhost:8001/v1
#   bash scripts/local_serve.sh qwen35_4b            # vLLM, http://localhost:8002/v1
#   bash scripts/local_serve.sh qwen35_4b_base       # vLLM, http://localhost:8003/v1
#   bash scripts/local_serve.sh clef_flash           # System One server, http://localhost:30000/v1/systemone
#   bash scripts/local_serve.sh pplx_decider_flash   # System One server, http://localhost:8010/v1/systemone
#   bash scripts/local_serve.sh kev4b                # Kev repository server, http://localhost:8008/v1/systemone
#
# Every model runs in bfloat16 on one CUDA GPU, at the revision pinned in its config.
# Environments (requirements/local-servers/): vllm.lock.txt (vLLM 0.33.0), dm.lock.txt (Clef-flash),
# pplx.lock.txt (Perplexity Decider), kev.lock.txt (Kev-4B, inside a clone of the Kev repository).
# Point VLLM, CLEF_PY, PPLX_PY and KEV_DIR at them (defaults: the current PATH / ./kev).
#
# vLLM: --max-num-seqs 8 = the runner's concurrency (8 requests in flight); --reasoning-parser qwen3 returns any
# <think> block separately (thinking is off in every request, chat_template_kwargs.enable_thinking=false).
# GPU_MEM_UTIL / KV_BYTES only size the server's memory; they do not change the outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM=${VLLM:-vllm}
CLEF_PY=${CLEF_PY:-python}
PPLX_PY=${PPLX_PY:-python}
KEV_DIR=${KEV_DIR:-kev}
MAX_SEQS=${MAX_SEQS:-8}

vllm_serve() {  # model revision port max-model-len [extra vllm args...]
  local model=$1 rev=$2 port=$3 len=$4; shift 4
  local mem=()
  [ -n "${GPU_MEM_UTIL:-}" ] && mem+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
  [ -n "${KV_BYTES:-}" ] && mem+=(--kv-cache-memory-bytes "$KV_BYTES")
  exec "$VLLM" serve "$model" --revision "$rev" --port "$port" --dtype bfloat16 \
    --max-num-seqs "$MAX_SEQS" --max-model-len "$len" ${mem[@]+"${mem[@]}"} "$@"
}

case "${1:-}" in
  qwen35_9b)      vllm_serve Qwen/Qwen3.5-9B      c202236235762e1c871ad0ccb60c8ee5ba337b9a 8001 34816 --reasoning-parser qwen3 ;;
  qwen35_4b)      vllm_serve Qwen/Qwen3.5-4B      851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 8002 34816 --reasoning-parser qwen3 ;;
  qwen35_4b_base) vllm_serve Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b 8003 8192 ;;
  clef_flash)
    exec "$CLEF_PY" scripts/serve_clef_flash.py --revision 22b75507981345858dd152fbd9a6f8c0c4068b0a --port 30000 ;;
  pplx_decider_flash)
    exec "$PPLX_PY" scripts/serve_pplx_decider.py --revision 9ff7cf9838e59df461c276b7699d56b244d2af58 --port 8010 ;;
  kev4b)  # tag v1.0 = revision 6cfce5c2fa4b4bd64026336ab649c5ca78857d52; kev.serve loads bfloat16 on a GPU
    cd "$KEV_DIR" && exec .venv/bin/python -m kev.serve --run jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --port 8008 ;;
  *) echo "usage: bash scripts/local_serve.sh {qwen35_9b|qwen35_4b|qwen35_4b_base|clef_flash|pplx_decider_flash|kev4b}" >&2; exit 1 ;;
esac
