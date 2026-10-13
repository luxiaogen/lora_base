#!/usr/bin/env bash
# 固定执行回归→Task0–1短测→一次完整T10；任何训练失败均停止，不重试。
set -euo pipefail
cd "$(dirname "$0")/.."
B_REPRO_PYTHON="${B_REPRO_PYTHON:-python}"
B_REPRO_ROOT="${1:-logs/repro_3090_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
: "${DUALMASK_REFERENCE:?需要设置旧B源码归档目录DUALMASK_REFERENCE}"
: "${EXPECTED_REVISION:?需要设置已推送的完整提交EXPECTED_REVISION}"
: "${PREDECESSOR_RESULT:?需要设置前序完整实验记录PREDECESSOR_RESULT}"
test "$(git rev-parse HEAD)" = "$EXPECTED_REVISION"
test -z "$(git status --porcelain)"
test ! -e "$B_REPRO_ROOT"
mkdir -p "$B_REPRO_ROOT"
export PYTHONDONTWRITEBYTECODE=1

record_state() {
    "$B_REPRO_PYTHON" -c 'import json, pathlib, sys
root, phase, child, queue, status = sys.argv[1:]
value = dict(queue_pid=int(queue), phase=phase, training_pid=int(child) if child else None, status=status)
for name in ("queue.json", "active.json"):
    target = pathlib.Path(root) / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(target)' "$B_REPRO_ROOT" "$1" "${2:-}" "$$" "$3"
}
on_exit() {
    code=$?
    if test "$code" -ne 0; then
        record_state "${phase:-preflight}" "" failed
    fi
}
trap on_exit EXIT
phase=preflight
record_state "$phase" "" running
"$B_REPRO_PYTHON" -c 'import json, pathlib, os, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "completed" and r["tasks_completed"] == 10 and r["planned_tasks"] == 10, "前序实验没有完整成功，停止接续"
pid = os.environ.get("PREDECESSOR_PID")
assert not pid or not pathlib.Path("/proc", pid).exists(), "前序训练进程仍然存在，停止接续"' "$PREDECESSOR_RESULT"
for file in models/attention.py models/losses.py methods/dlora.py utils/depth_permissions.py utils/toolkit.py reference_run.json; do
    test -f "$DUALMASK_REFERENCE/$file"
done

# 默认不与用户的Python GPU进程并行；只有用户明确授权的暂停调试进程可驻留。
"$B_REPRO_PYTHON" -c 'import os, pathlib, subprocess, time, torch
assert torch.cuda.is_available(), "CUDA不可用"
assert "3090" in torch.cuda.get_device_name(0), "不是授权的3090"
for sample in range(2):
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    for row in rows.splitlines():
        pid, command = row.split(",", 1)
        if int(pid) == os.getpid(): continue
        if "python" not in command.lower(): continue
        try: full = pathlib.Path("/proc", pid.strip(), "cmdline").read_text()
        except OSError: continue
        allowed = os.environ.get("ALLOW_IDLE_DEBUGGER_PID")
        assert allowed == pid.strip() and "pydevd.py" in full, "仍有训练或未授权调试进程：" + row
    use = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    assert int(use.strip()) <= 15, "GPU仍在计算，停止接续"
    if sample == 0: time.sleep(3)
print("3090就绪；不修改任何原项目文件")'

phase=tests
record_state "$phase" "" running
"$B_REPRO_PYTHON" -m unittest discover -s test -v > "$B_REPRO_ROOT/test.log" 2>&1
phase=smoke
"$B_REPRO_PYTHON" main.py --config exps/dlora/imgr10.json --epochs 1 --ca-epochs 1 --max-tasks 2 \
    --output "$B_REPRO_ROOT/smoke" > "$B_REPRO_ROOT/smoke.stdout.log" 2>&1 &
child=$!
record_state "$phase" "$child" running
echo "短测：queue PID=$$，training PID=$child，目录=$B_REPRO_ROOT/smoke"
wait "$child"
"$B_REPRO_PYTHON" scripts/check_reproduction.py "$B_REPRO_ROOT/smoke" --tasks 2 --epochs 1
phase=formal
"$B_REPRO_PYTHON" main.py --config exps/dlora/imgr10.json --output "$B_REPRO_ROOT/formal" \
    > "$B_REPRO_ROOT/formal.stdout.log" 2>&1 &
child=$!
record_state "$phase" "$child" running
echo "正式T10：queue PID=$$，training PID=$child，目录=$B_REPRO_ROOT/formal"
wait "$child"
phase=analysis
record_state "$phase" "" running
"$B_REPRO_PYTHON" scripts/check_reproduction.py "$B_REPRO_ROOT/formal" > "$B_REPRO_ROOT/comparison.log" 2>&1
record_state completed "" completed
echo "复现验证结束：$B_REPRO_ROOT/formal/comparison.json"
