#!/bin/bash
# Re-run of the mount check with the 2026-11-17 fix (finger links read from the physics view), after the v5 pilot.
# v4 first as the negative control: its 180 deg shift must be >= 100 mm, else the measurement is void; then v5 (pass <= 2 mm).
#   setsid nohup scripts/mount_check_fixed_driver.sh <PID to wait for> > eval_runs/mount_check_fixed/driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
WAIT_PID="${1:-}"
OUT=eval_runs/mount_check_fixed
RES=eval_results/mount_check_fixed
mkdir -p "$OUT" "$RES"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
if [ -n "$WAIT_PID" ]; then
    say "waiting for PID $WAIT_PID (v5 pilot driver)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
sleep 30
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then say "GPU job running -- not starting"; exit 1; fi
for prof in v4 v5; do
    say "mount check $prof start"
    setsid timeout 1800 env VLA_SIM_PROFILE=$prof GPU=0 scripts/isaac_py.sh render_mount_check.py --out "$PWD/$RES/$prof" \
        > "$OUT/mount_check_$prof.log" 2>&1 &
    p=$!; wait $p; kill -- -$p 2>/dev/null
    say "mount check $prof end: $(grep -m1 '^VERDICT' "$OUT/mount_check_$prof.log" || echo 'no verdict -- see log')"
done
python3 - "$RES" <<'EOF' | tee "$OUT/verdict.log"
import json, os, sys
res = sys.argv[1]
def load(p):
    f = os.path.join(res, p, "mount_check.json")
    return json.load(open(f))["verdict"] if os.path.exists(f) else None
v4, v5 = load("v4"), load("v5")
valid = bool(v4 and "measured_fingertip_shift_mm_at_90_180" in v4 and v4["measured_fingertip_shift_mm_at_90_180"][1] >= 100)
out = dict(v4=v4, v5=v5, measurement_valid_on_v4=valid,
           v5_coaxial_pass=(bool(v5.get("coaxial_pass")) if valid and v5 and "coaxial_pass" in v5 else None))
json.dump(out, open(os.path.join(res, "verdict.json"), "w"), indent=1)
print("VERDICT", json.dumps(out))
EOF
say "mount check (fixed) finished"
