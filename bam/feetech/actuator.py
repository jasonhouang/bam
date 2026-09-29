# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Union
from bam.message import yellow, print_parameter, bright
from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench, Pendulum

if TYPE_CHECKING:
    from bam.actuator import ArrayLike


class STS3215Actuator(VoltageControlledActuator):
    """
    Feetech STS3215 7.4v
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            # This gain, if multiplied by a position error and firmware KP, gives duty cycle
            # It was determined using an oscilloscope and STS3215 actuators
            # here, firmware_kp = kp
            error_gain=0.166,
            # self.error_gain = 0.001 * np.rad2deg(1.0)
            # Maximum allowable duty cycle, also determined with oscilloscope
            max_pwm=0.97,  # TODO, but can we assume 1.0 ?
        )

        self.default_max_velocity = (3400 * 2 * np.pi) / 4096

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def load_log(self, log: dict):
        super().load_log(log)

        self.q_target_smooth = np.zeros_like(self.kp)

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)]
        self.model.kt = Parameter(0.784532, 0.05, 2.5)  # docs says 8 kg.cm / A

        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

        # Motor resistance [Ohm]
        self.model.R = Parameter(2.0, 0.1, 10.0)

        # Motor armature / apparent inertia [kg m^2]
        self.model.armature = Parameter(0.0001, 0.00001, 0.04)

        self.model.q_offset = Parameter(0, -0.2, 0.2)

        self.model.max_velocity = Parameter(
            self.default_max_velocity,
            0.1 * self.default_max_velocity,
            10.0 * self.default_max_velocity,
        )

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """
        Assumes the motor is using a kp controller
        This can be overloaded if more custom behaviour is used
        """

        # Internal target position is clipped using maximum velocity
        self.q_target_smooth = self.backend.clamp(
            q_target,
            self.q_target_smooth - self.model.max_velocity.value * dt,
            self.q_target_smooth + self.model.max_velocity.value * dt,
        )

        duty_cycle = (
            (self.q_target_smooth - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)

        return self.vin * duty_cycle


class HD1910Actuator(VoltageControlledActuator):
    """
    Feetech HD-1910 5V (12 kg.cm, TTL serial).

    Same firmware rate-limiting P-controller architecture as the STS3215
    (compute_control rate-limits the internal target via max_velocity), but
    with different motor constants identified on the pendulum bench at
    5.0-5.2 V. Powered by a regulated 5.0 V rail; never 12 V on this servo.
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=5.0,
            kp=32,        # HD-1910 firmware Kp (reg50 SRAM / reg21 EPROM = 32)
            error_gain=0.163,
            max_pwm=0.97,
        )
        self.default_max_velocity = 8.0282  # rad/s, from bench identification
        # Lazily initialised on first compute_control (or in load_log / reset)
        # so that both the pendulum bench and the warp training path work
        # without an explicit seeding step.
        self.q_target_smooth = None

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def load_log(self, log: dict):
        super().load_log(log)
        self.q_target_smooth = np.zeros_like(self.kp)

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)] — bench-identified
        self.model.kt = Parameter(0.692, 0.05, 2.5)

        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

        # Motor resistance [Ohm] — bench-identified
        self.model.R = Parameter(3.75, 0.1, 10.0)

        # Motor armature / apparent inertia [kg m^2] — bench-identified
        self.model.armature = Parameter(0.00224, 0.0001, 0.04)

        self.model.q_offset = Parameter(0, -0.2, 0.2)

        self.model.max_velocity = Parameter(
            self.default_max_velocity,
            0.1 * self.default_max_velocity,
            10.0 * self.default_max_velocity,
        )

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        # Lazy init: on the very first call (warp training path, or CPU
        # inference before reset_bam_ctrl), seed from the current position so
        # the rate-limiter starts at equilibrium instead of zero.
        if self.q_target_smooth is None:
            self.q_target_smooth = q
        # Internal target position is clipped using maximum velocity
        self.q_target_smooth = self.backend.clamp(
            q_target,
            self.q_target_smooth - self.model.max_velocity.value * dt,
            self.q_target_smooth + self.model.max_velocity.value * dt,
        )

        duty_cycle = (
            (self.q_target_smooth - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)

        return self.vin * duty_cycle

    def reset(self, env_ids=...) -> None:
        """Clear the rate-limiter state so it re-seeds on the next compute_control."""
        self.q_target_smooth = None
