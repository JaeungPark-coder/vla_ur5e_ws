import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("summarize_eval", Path(__file__).resolve().parents[1] / "scripts" / "summarize_eval.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)


def test_wilson_known_values():
    lo, hi = se.wilson(10, 20)
    assert abs(lo - 0.299) < 0.002 and abs(hi - 0.701) < 0.002
    assert se.wilson(0, 20)[0] == 0.0 and se.wilson(20, 20)[1] == 1.0


def test_summary_pairs_and_blowups_separately(tmp_path):
    csv_path = tmp_path / "r.csv"
    lines = ["trial,success,steps,xy_error_mm,max_cube_z_sampled,max_cube_z_per_tick,blowup,execute_horizon"]
    ks = [10, 25, 50]
    succ = [True, False, False, True, True, False]
    for i in range(6):
        lines.append(f"{i + 1},{succ[i]},900,5.0,0.1,0.16,0,{ks[i % 3]}")
    csv_path.write_text("\n".join(lines))
    (tmp_path / "trial_002.json").write_text(json.dumps({"blowup": True, "reasons": {"jump": 1}}))
    rows = se.load(str(csv_path), str(tmp_path))
    s = se.summarize(rows)
    assert s[10]["success"] == 2 and s[25]["blowups"] == 1 and s[25]["n_no_blowup"] == 1 and s[50]["success"] == 0
    p = se.paired(rows, 3)
    assert p[0] == {10: True, 25: False, 50: False} and p[1] == {10: True, 25: True, 50: False}
