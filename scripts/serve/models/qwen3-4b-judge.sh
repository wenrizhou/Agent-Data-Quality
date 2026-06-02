#!/bin/bash
# SGLang judge model preset.
#
# Copy this file for a new model and update MODEL / MODEL_NAME / parallelism.

MODEL="${MODEL:-/work/projects/polyullm/yiming/slime-workspace/models/Qwen3-4B/}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b-judge}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-31877}"

TP="${TP:-1}"
DP="${DP:-1}"

WARMUPS="${WARMUPS:-3}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

# Useful for Qwen-style tool call parsing. Leave empty to omit the argument.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"

# Optional SGLang flags. Leave empty to omit.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Runtime paths.
LOG_DIR="${LOG_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
