"""Request/acknowledge bookkeeping for pick_place_scene_bridge.py's lockstep
mode -- pure Python, no Isaac Sim or ROS import, so it is unit-testable
(test/test_lockstep_protocol.py).

Why lockstep exists (found 2026-10-05, code reading only): by default the
bridge free-runs `world.step(render=True)` with no clock, while the policy
client blocks on `policy.infer()` for every action. One action therefore
spans an unknown number N of sim ticks, but the demonstrations are exactly
ONE action per tick (collect_demos.py logs every physics tick; tool motion is
only ~1-4 mrad per tick). Executed that way the arm covers ~1/N of the
demonstrated motion per sim-second, and `move_joints` returns immediately
because its 0.02 rad tolerance is far larger than one tick's motion.

Lockstep removes the ambiguity: the client sends request k, the bridge applies
it and steps the simulator EXACTLY ONE tick, then publishes the new observation
stamped with k. Inference latency no longer matters to sim time.

Wire format (sensor_msgs/JointState on /vla/joint_target):
  header.stamp.sec   request id k (> 0, strictly increasing)
  position[0:6]      arm joint targets
  position[6]        gripper command 0..1 (optional; falls back to the
                     separate gripper topic when absent)
and every observation the bridge publishes in lockstep mode carries the id of
the last request it executed in header.stamp.sec (0 before the first one).
"""


class LockstepGate:
    def __init__(self):
        self.last_executed_id = 0
        self._pending = None  # (id, joints, gripper) -- only the newest is kept

    def submit(self, request_id, joints, gripper=None):
        """Called from the subscription callback. Returns True if the request
        is new; stale or repeated ids (a DDS redelivery, a request older than
        what was already executed) are ignored so a tick is never run twice."""
        request_id = int(request_id)
        if request_id <= self.last_executed_id:
            return False
        if self._pending is not None and request_id <= self._pending[0]:
            return False
        self._pending = (request_id, joints, gripper)
        return True

    def take(self):
        """Called from the sim loop. Returns (id, joints, gripper) when there
        is a request to execute -- the caller must then step exactly one
        tick -- else None. Marks it executed."""
        if self._pending is None:
            return None
        request = self._pending
        self._pending = None
        self.last_executed_id = request[0]
        return request

    def reset(self):
        """Scene reset. The ids keep counting up on the client side, so the
        executed id is NOT rewound; only the queued request is dropped."""
        self._pending = None
