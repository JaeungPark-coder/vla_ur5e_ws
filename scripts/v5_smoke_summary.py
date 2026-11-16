#!/usr/bin/env python3
"""Reads the v5 expert smoke run (scripts/v5_smoke_driver.sh) against the verdicts set in README before it ran:
success (saved / attempts) >= 80 %, blow-ups <= 10 % of attempts, v5_expert_check PASS; per region as well.
    python3 scripts/v5_smoke_summary.py <log dir> <expert check json> <mount check json> --out summary.json
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument('logdir')
ap.add_argument('check_json')
ap.add_argument('mount_json')
ap.add_argument('--out', required=True)
a = ap.parse_args()

regions, tot = {}, dict(attempts=0, saved=0, blowups=0, never_lifted=0, placed_off=0, other_rejects=0)
for log in sorted(glob.glob(os.path.join(a.logdir, 'v5smoke_*.log'))):
    name = os.path.basename(log)[len('v5smoke_'):-4]
    r = dict(attempts=0, saved=0, blowups=0, never_lifted=0, placed_off=0, other_rejects=0, finished=False)
    for line in open(log, errors='replace'):
        if line.startswith('attempt '):
            r['attempts'] += 1
            if ': OK' in line:
                r['saved'] += 1
            elif 'blow-up' in line:
                r['blowups'] += 1
            elif 'never lifted' in line:
                r['never_lifted'] += 1
            elif 'off target' in line:
                r['placed_off'] += 1
            else:
                r['other_rejects'] += 1
        if line.startswith('done -- dataset written'):
            r['finished'] = True
    r['success_rate'] = r['saved'] / r['attempts'] if r['attempts'] else None
    regions[name] = r
    for k in tot:
        tot[k] += r[k]
tot['success_rate'] = tot['saved'] / tot['attempts'] if tot['attempts'] else None
tot['blowup_rate'] = tot['blowups'] / tot['attempts'] if tot['attempts'] else None
check = json.load(open(a.check_json))['summary'] if os.path.exists(a.check_json) else None
mount = json.load(open(a.mount_json)).get('verdict') if os.path.exists(a.mount_json) else None
verdict = dict(
    mount_coaxial=None if not mount or 'coaxial_pass' not in mount else mount['coaxial_pass'],
    expert_success_ge_80=None if tot['success_rate'] is None else tot['success_rate'] >= 0.80,
    blowups_le_10=None if tot['blowup_rate'] is None else tot['blowup_rate'] <= 0.10,
    expert_check_pass=None if check is None else check['PASS'])
out = dict(total=tot, regions=regions, expert_check=check, mount_check=mount, verdict=verdict,
           smoke_pass=all(v is True for k, v in verdict.items() if k != 'mount_coaxial'))
print(json.dumps(out, indent=1))
with open(a.out, 'w') as f:
    json.dump(out, f, indent=1)
