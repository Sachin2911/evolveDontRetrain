"""Benchmark scoring and continuation, independent of simulator reset behaviour."""


class EpisodeScore:
    def __init__(self, continuation, horizon=300, extension=40, dt=0.05):
        self.continuation = continuation
        self.horizon, self.extension, self.dt = horizon, extension, dt
        self.controls = 0
        self.benchmark = None
        self.extension_steps = 0
        self.predicate_lost = False
        self.post_success_violation = False
        self.finished = False

    def advance(self, *, success, physical_failure, boundary):
        if self.finished:
            raise RuntimeError("Cannot advance a completed episode")
        self.controls += 1
        violation = physical_failure or boundary
        if self.benchmark is None:
            outcome = (
                "physical_failure"
                if physical_failure
                else "boundary_exit"
                if boundary
                else "success"
                if success
                else "timeout"
                if self.controls >= self.horizon
                else None
            )
            if outcome is not None:
                self.benchmark = {
                    "outcome": outcome,
                    "safe_success": outcome == "success",
                    "violation": violation,
                    "physical_failure": physical_failure,
                    "boundary_exit": boundary,
                    "timeout": outcome == "timeout",
                    "steps": self.controls,
                    "completion_time_s": self.controls * self.dt if outcome == "success" else None,
                    "J": self.controls * self.dt
                    if outcome == "success"
                    else self.horizon * self.dt,
                }
                self.finished = outcome != "success" or not self.continuation
        else:
            self.extension_steps += 1
            self.predicate_lost |= not success
            self.post_success_violation |= violation
            self.finished = violation or self.extension_steps >= self.extension
        return self.finished

    def summary(self):
        if not self.finished:
            raise RuntimeError("An unfinished attempt has no valid episode summary")
        return {
            **self.benchmark,
            "controls": self.controls,
            "continuation_enabled": self.continuation,
            "continuation_steps": self.extension_steps,
            "sustained_safe_success": (
                self.benchmark["safe_success"]
                and self.continuation
                and self.extension_steps == self.extension
                and not self.predicate_lost
                and not self.post_success_violation
            )
            if self.continuation
            else None,
            "post_success_predicate_loss": self.predicate_lost if self.continuation else None,
            "post_success_violation": self.post_success_violation if self.continuation else None,
        }
