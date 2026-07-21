#!/usr/bin/env bash
# fastserve installer — one command, everything set up.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

echo "== fastserve installer =="

command -v python3 >/dev/null 2>&1 || { echo "python3 not found. Install Python 3.9+ first."; exit 1; }

if [ ! -d "$VENV" ]; then
  echo "-> creating virtual environment at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip -q

echo "-> installing vLLM (pulls in a matching torch/CUDA build automatically)"
pip install vllm -q

echo "-> installing fastserve"
pip install -e "$HERE" -q

# Shim so the user never has to remember to activate the venv.
cat > "$HERE/fastserve" <<SHIM
#!/usr/bin/env bash
exec "$VENV/bin/fastserve" "\$@"
SHIM
chmod +x "$HERE/fastserve"

echo ""
echo "done. try it now:"
echo ""
echo "   ./fastserve info  Qwen/Qwen3-8B"
echo "   ./fastserve bench Qwen/Qwen3-8B"
echo "   ./fastserve serve Qwen/Qwen3-8B"
echo ""
