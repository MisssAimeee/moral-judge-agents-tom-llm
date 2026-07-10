#!/usr/bin/env bash
# setup_api_models.sh -- Install API client packages for closed-model providers.
# Run on the LOGIN node (has internet). GPU not needed for API models.
#
#   bash engaging/setup_api_models.sh
#
# After this, set your API keys and run:
#   bash engaging/run_api_models.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

pip install --upgrade \
    openai \
    anthropic \
    google-generativeai \
    mistralai \
    tiktoken

echo ""
echo "Installed API packages. Versions:"
python - <<'PY'
import importlib, pkg_resources
for pkg in ["openai","anthropic","google.generativeai","mistralai"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", pkg_resources.get_distribution(pkg.split(".")[0]).version)
        print(f"  {pkg:<25s}  {v}")
    except Exception as e:
        print(f"  {pkg:<25s}  MISSING ({e})")
PY

echo ""
echo "Next steps:"
echo "  1. Set API keys in your shell (or put them in a .env file NOT committed to git):"
echo "       export OPENAI_API_KEY='sk-...'"
echo "       export ANTHROPIC_API_KEY='sk-ant-...'"
echo "       export GOOGLE_API_KEY='AIza...'"
echo "       export MISTRAL_API_KEY='...'"
echo "       export TOGETHER_API_KEY='...'"
echo ""
echo "  2. Smoke-test one API model (small/cheap):"
echo "       python code/03_behavioral.py --backend openai --models gpt-4o-mini --n_samples 2 --limit 10"
echo ""
echo "  3. Full API run (after current GPU job finishes):"
echo "       bash engaging/run_api_models.sh"
