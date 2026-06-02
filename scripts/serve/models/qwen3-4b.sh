#!/bin/bash
# Template for a SGLang model preset used by data_quality/scripts/serve.

MODEL="${MODEL:-/work/projects/polyullm/yuhang/post-train/models/Qwen/Qwen3-4B/}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-31877}"

TP="${TP:-1}"
DP="${DP:-1}"

WARMUPS="${WARMUPS:-3}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"

LOG_DIR="${LOG_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
