"""Local observed-BEFORE producer composition; actual launch is not admitted.

A fixed executable description is integrity data, never a launch permission.
The existing outer runtime cannot select this module. Its command-line entry
refuses until the separately reviewed parent/bootstrap and tagged consumers exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

from scripts import strength_freshness_cpu_before_measurement as measurement
from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_outer as outer
from scripts import strength_freshness_cpu_window_contract as bodies

MODULE = "scripts.strength_freshness_cpu_observed_capture_cli"
ACTUAL_LAUNCH_REFUSAL = "observed-launcher-composition-not-qualified"
POLL_SECONDS = 5.0


class ProducerRefusal(ValueError):
    """Fixed private-data-free local composition failure."""


def _remaining(window, last):
    deadline = window["deadline"]
    now = {
        "monotonic_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
    }
    if any(type(value) is not int or value <= 0 for value in now.values()):
        raise ProducerRefusal("observed-producer-clock-type")
    if any(now[axis] < last[axis] for axis in now):
        raise ProducerRefusal("observed-producer-clock-regression")
    last.update(now)
    remaining = min(deadline[axis] - now[axis] for axis in now)
    if remaining <= 0:
        raise ProducerRefusal("observed-producer-original-deadline")
    return remaining


def _join_observation(window, last, start, end):
    """Join actual retained IO brackets to control samples, without new IO."""
    completion.clock(start)
    completion.clock(end)
    completion.order(last, start)
    completion.order(start, end)
    completion.limit(end, window["deadline"], strict=True)
    last.update(end)


def collect_before_artifacts(
    registration,
    io,
    *,
    expected_window,
    external_premises,
    verified_champions,
    bindings,
    expected,
    output_root,
):
    """Own the complete local measurement; return bytes, never publish or admit.

    The external writer/source/access premises remain the caller's independent
    responsibility. This API is unavailable through main and accepts no passed
    measurement or proof. Local positive fixtures are not executable provenance.
    """
    try:
        bodies.checked_expected(expected, registration)
        if not (
            expected_window["phase"] == "before"
            and bodies.same(expected["bindings"], bindings)
            and bodies.same(expected["expected_window"], expected_window)
            and bodies.same(expected["verified_champions"], verified_champions)
        ):
            raise ProducerRefusal("observed-producer-input-binding")
        last = dict(expected_window["phase_start"])
        _remaining(expected_window, last)
        session = measurement.BeforeMeasurementSession(
            registration,
            io,
            expected_window=expected_window,
            external_premises=external_premises,
            verified_champions=verified_champions,
            bindings=bindings,
        )
        pending = session.begin()
        _join_observation(
            expected_window, last, pending["read_start"], pending["read_end"]
        )
        _remaining(expected_window, last)
        # Reserve the existing final publication round for finish(). Neither
        # this loop nor a normal UTD wait can renew IO or choose a new endpoint.
        for _ in range(255):
            _remaining(expected_window, last)
            pending = session.observe()
            _join_observation(
                expected_window, last, pending["read_start"], pending["read_end"]
            )
            _remaining(expected_window, last)
            if pending["renewal_complete"]:
                break
            time.sleep(min(POLL_SECONDS, _remaining(expected_window, last) / 10**9))
        else:
            raise ProducerRefusal("observed-producer-renewal-incomplete")
        measured = session.finish().private_copy()
        capture = measured["capture_value"]
        _join_observation(
            expected_window,
            last,
            capture["metric_window"]["end_fence"]["fence"]["audit"]["read_start"],
            capture["read_end"],
        )
        _remaining(expected_window, last)
        result = bodies.encode_before_bodies(
            measured["capture_value"],
            measured["provenance_value_without_capture_pin"],
            expected=expected,
            registration=registration,
            output_root=output_root,
        )
        _remaining(expected_window, last)
        return result
    except ProducerRefusal:
        raise
    except Exception:
        # Raw producer rows, argv, environments and inner exception messages
        # are private. Actual process/source admission remains absent here.
        raise ProducerRefusal("observed-producer-incomplete-or-refused") from None


@dataclass(frozen=True)
class ObservedCaptureSpec:
    """One fixed, BEFORE-only contract, unselected by the old runtime."""

    python: str
    control_root: str
    launch_path: str
    launch_sha256: str
    deadline_monotonic_ns: int

    def argv(self) -> tuple[str, ...]:
        # Reuse the unchanged path/deadline checks, without a caller-selected
        # module or subclass that an old runtime could select as CaptureSpec.
        old = outer.CaptureSpec(
            "before",
            self.python,
            self.control_root,
            self.launch_path,
            self.launch_sha256,
            self.deadline_monotonic_ns,
        ).argv()
        return (*old[:5], MODULE, *old[6:])

    def contract(self) -> dict[str, Any]:
        return {
            "argv": self.argv(),
            "cwd": self.control_root,
            "env": dict(outer.ENV),
            "resources": dict(outer.CAPTURE_LIMITS),
        }


def observed_collector_contract_sha256(
    python, control_root, launch_pin, original_deadline_ns
):
    completion.pin(launch_pin)
    spec = ObservedCaptureSpec(
        python,
        control_root,
        launch_pin["path"],
        launch_pin["sha256"],
        original_deadline_ns,
    )
    return outer.digest(spec.contract())


def main(argv=None):
    # Deliberately do not parse a launch, inspect files, initialize a runtime,
    # invoke a callback, or permit JSON/flags to turn this refusal into admission.
    del argv
    print(ACTUAL_LAUNCH_REFUSAL, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
