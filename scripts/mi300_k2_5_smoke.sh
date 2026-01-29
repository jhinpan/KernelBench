#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL_ID="${MODEL_ID:-/data/Kimi-K2.5}"

echo "[MI300-K2.5] Checking server: ${BASE_URL}"

tmp_json="$(mktemp)"
trap 'rm -f "${tmp_json}"' EXIT

curl -fsS "${BASE_URL}/v1/models" -o "${tmp_json}"
python - <<'PY' "${tmp_json}" "${MODEL_ID}"
import json, sys
path = sys.argv[1]
model = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ids = [m.get("id") for m in data.get("data", [])]
print("[MI300-K2.5] /v1/models:", ids)
if model not in ids:
    raise SystemExit(f"Expected model '{model}' not found in /v1/models")
PY

echo "[MI300-K2.5] Chat completion smoke test"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly OK\"}],\"max_tokens\":16,\"temperature\":0}" \
  -o "${tmp_json}"
python - <<'PY' "${tmp_json}"
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
msg = data["choices"][0]["message"]["content"]
print("[MI300-K2.5] Response:", msg.strip())
PY

echo "[MI300-K2.5] OK"
