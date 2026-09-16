#!/usr/bin/env bash
set -uo pipefail

[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
python -m unittest test.test_task_consistency_scripts test.test_predicted_onehot test.test_p_conflict_diagnostics || exit 1
for config in exps/dlora/imgr10.json exps/dlora/imgr20.json; do
    python -c 'import json, pathlib, sys; p=pathlib.Path(json.load(open(sys.argv[1]))["data_path"]); print(sys.argv[1], "->", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"' "$config" || exit 1
done
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short

FAILED=0
bash scripts/9_17_task_consistency_imgr10_3090.sh || FAILED=1
bash scripts/9_17_task_consistency_imgr20_3090.sh || FAILED=1
exit "$FAILED"
