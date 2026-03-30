#!/usr/bin/env bash
# =============================================================================
# Switch from pip-installed hermes-agent to an editable install from this
# git clone. Matches what `hermes update` expects (git + pip install -e).
#
# Always uses venv/bin/python -m pip (never bare `pip`) so Debian/Ubuntu
# PEP 668 "externally-managed-environment" does not trigger when the shell
# picks up system pip by mistake.
#
# Usage (from repo root):
#   bash scripts/switch-to-editable-git.sh
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VPY="$ROOT/venv/bin/python"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "✗ No .git directory in: $ROOT"
  echo "  This folder must be a git clone. For example:"
  echo "    git clone https://github.com/NousResearch/hermes-agent.git"
  echo "    cd hermes-agent"
  echo "    bash scripts/switch-to-editable-git.sh"
  exit 1
fi

if [[ ! -d "$ROOT/venv" ]]; then
  echo "→ Creating venv at $ROOT/venv ..."
  python3 -m venv "$ROOT/venv"
fi

if [[ ! -x "$VPY" ]]; then
  echo "✗ Missing or non-executable: $VPY"
  echo "  Remove the broken venv and re-run:"
  echo "    rm -rf \"$ROOT/venv\" && bash scripts/switch-to-editable-git.sh"
  exit 1
fi

# Ensure pip exists inside the venv (some minimal images skip it).
if ! "$VPY" -m pip --version &>/dev/null; then
  echo "→ Bootstrapping pip in venv (ensurepip) ..."
  "$VPY" -m ensurepip --upgrade --default-pip
fi

echo "→ Using interpreter: $($VPY -c 'import sys; print(sys.executable)')"

echo "→ Uninstalling any previously pip-installed hermes-agent in this venv (safe if none) ..."
"$VPY" -m pip uninstall -y hermes-agent 2>/dev/null || true

echo "→ pip install -e \".[all]\" (editable, same as hermes update) ..."
if ! "$VPY" -m pip install -e ".[all]" ; then
  echo "  ⚠ .[all] failed; trying .[vertex] ..."
  if ! "$VPY" -m pip install -e ".[vertex]" ; then
    echo "  ⚠ .[vertex] failed; installing base package only ..."
    "$VPY" -m pip install -e "."
  fi
fi

echo ""
echo "→ Where does \`hermes\` load from (via this venv's python)?"
"$VPY" -c "import hermes_cli.main as m; print(m.PROJECT_ROOT.resolve())"

echo ""
echo "✓ Editable install ready."
echo ""
echo "  Always use the venv interpreter (avoids system pip / PEP 668):"
echo "    source \"$ROOT/venv/bin/activate\""
echo "    which python   # should be $ROOT/venv/bin/python"
echo "    python -m pip --version"
echo "    which hermes && hermes --version"
echo ""
echo "  Vertex Gemini also needs:"
echo "    \"$VPY\" -m pip install google-genai"
echo "  Stay current:"
echo "    cd \"$ROOT\" && source venv/bin/activate && hermes update"
echo ""
