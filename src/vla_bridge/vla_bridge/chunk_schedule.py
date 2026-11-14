"""Which action is applied on each tick of one control iteration when inference latency is emulated in TICKS (lockstep kept,
so the same seed gives the same scene and runs can be paired). README "inference latency in simulation".

One iteration = one observation + one policy call. d = latency in ticks, K = execute_horizon.
  sync  : the robot waits for the answer: d ticks holding the last target, then rows 0..K-1 of the new chunk. K + d ticks per
          iteration (d = 0 is the original behaviour).
  async : the robot keeps executing the previous chunk while the answer is "in flight": rows prev_pos..prev_pos+d-1 of the
          previous chunk, then the new chunk from row d (its first d rows describe ticks that have already passed). K ticks per
          iteration, each chunk applied at rows d..d+K-1, so K + d <= chunk length is needed. The first chunk of a trial has no
          previous chunk: the robot stands (d holds) and the chunk starts at row 0, as a real robot starting from rest would.
          d = 0 is the original behaviour.
"""

HOLD, PREV, NEW = "hold", "prev", "new"


def plan_iteration(mode, K, d, chunk_len, prev_pos=None):
    """Returns (steps, next_prev_pos). steps: list of (source, row) with source in {HOLD, PREV, NEW} (row is None for HOLD).
    prev_pos: next unused row of the previous chunk (None = no previous chunk, i.e. the first iteration of a trial).
    next_prev_pos: the value to pass as prev_pos next time (the new chunk becomes the previous one)."""
    if mode not in ("sync", "async"):
        raise ValueError(f"unknown delay mode {mode!r}")
    K, d = int(K), int(d)
    if K < 1 or d < 0:
        raise ValueError("need K >= 1 and d >= 0")
    if d == 0:
        n = min(K, chunk_len)
        return [(NEW, i) for i in range(n)], n
    if mode == "sync":
        n = min(K, chunk_len)
        return [(HOLD, None)] * d + [(NEW, i) for i in range(n)], n
    # async
    if d >= K:
        raise ValueError(f"async needs d < K (d={d}, K={K})")
    if K + d > chunk_len:
        raise ValueError(f"async needs K + d <= chunk length (K={K}, d={d}, chunk={chunk_len})")
    if prev_pos is None:
        steps = [(HOLD, None)] * d + [(NEW, i) for i in range(K - d)]
        return steps, K - d
    steps = [(PREV, prev_pos + i) for i in range(d)] + [(NEW, d + i) for i in range(K - d)]
    return steps, K
