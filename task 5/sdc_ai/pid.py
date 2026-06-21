from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PIDController:
    """Simple PID controller with anti-windup and output clamping."""
    kp: float
    ki: float
    kd: float
    output_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    integral_limit: Optional[float] = None

    _integral: float = field(default=0.0, init=False)
    _prev_error: Optional[float] = field(default=None, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")

        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        low, high = self.output_limits
        if low is not None:
            output = max(low, output)
        if high is not None:
            output = min(high, output)
        return output
