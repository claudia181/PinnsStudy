"""
reaction_source.py
===========

This module implements the `Source` class, representing the source reaction function.

Spatio-temporal domain: xyt

Classes:
- `Source` (For creating explicit or implicit scalar source fields)
"""

import torch
import numpy as np
from typing import Callable, Tuple, Self

# ===================================== Source class =====================================
class Source:
    """
    Class for source functions involved in the reaction process.

    Maps vector-type object into scalars (producing a scalar field):
    - Callable[[np.ndarray, ..., np.ndarray], np.ndarray],
    - Callable[[torch.Tensor, ..., torch.Tensor], torch.Tensor].

    The input of the source function can be the spatio-temporal coordinates (or a subset of them) or the scalar field `u` (implicit sources).

    Attributes
    ----------
    mode : str
        The type of source: 'constant', 'decay', 'periodic', 'temporary', 'logistic', 'AllenCahn', 'Arrhenius'.
    sigma : float
        Std dev of the gaussian bump of the source.
    center : Tuple[float, float]
        Center of the gaussian bump of the source.
    amp : float
        Amplitude of the gaussian bump of the source.
    delta : float
        Decay factor for decaying sources.
    period : float
        Period for periodic sources.
    A : float
        Parameter for implicit sources 'logistic', 'AllenCahn' or 'Arrhenius'.
    B : float
        Parameter for implicit sources 'logistic' or 'Arrhenius'.
    fn : Callable
        The source function applied:
        - 'constant': s(x, y) = amp * G(x, y)
        - 'decay': s(x, y, t) = amp * e^(- delta * t) * G(x, y)
        - 'periodic': s(x, y, t) = amp * sin((2 pi / period) * t) * G(x, y)
        - 'temporary': s(x, y, t) = amp * (t < period) * G(x, y)
        - 'logistic': s(u) = A * u^2 - B * u
        - 'AllenCahn': s(u) = A * (u^3 - u)
        - 'Arrhenius': s(u) = A * e^(- B / u)
    """
    def __init__(
            self,
            mode: str,
            sigma: float = None,
            center: Tuple[float, float] = None,
            amp: float = None,
            delta: float = None,
            period: float = None,
            A: float = None,
            B: float = None
    ) -> None:
        """
        Constructor: build a scalar source function with state s(x, y, t, u) with values in R.
        
        Gaussian sources s(x, y, t) =
            - amp * f(t) * G(x, y)
            - G(x, y) = e^( -((x - xc)^2 + (y - yc)^2) / (2 * sigma^2) )
            - f(t) =
                - 1 (constant)
                - e^(- delta * t) (decay)
                - sin((2 pi / period) * t) (periodic)
                - (t < period) (temporary)
        Sources s(u) =
            - A * u^2 - B * u (logistic)
            - A * (u^3 - u) (AllenCahn)
            - A * e^(- B / u) (Arrhenius)
        
        Parameters
        ----------
        mode : str
            Source function type identifier:
            - "constant": s(x, y) = amp * G(x, y)
            - "decay": s(x, y, t) = amp * e^(- delta * t) * G(x, y)
            - "oscillate": s(x, y, t) = amp * sin((2 pi / period) * t) * G(x, y)
            - "temporary": s(x, y, t) = amp * (t < period) * G(x, y)
            - "logistic": s(u) = A * u^2 - B * u
            - "AllenCahn": s(u) = A * (u^3 - u)
            - "Arrhenius": s(u) = A * e^(- B / u)
        sigma : float
            Standard deviation of the Gaussian.
        center : tuple
            Center (xc, yc) of the Gaussian.
        amp : float
            For Gaussian-type sources: s(x, y, t) = amp * f(t) * G(x, y).
        delta : float
            For mode = "decay", the decay rate: f(t) = e^(- delta * t).
        period : float
            - For mode = "oscillate": f(t) = sin(2 * pi / period * t).
            - For mode = "temporary": f(t) = (t < period).
        A : float
            - For mode = "logistic": s(u) = A * u^2 - B * u
            - For mode = "AllenCahn": s(u) = A * (u^3 - u)
            - For mode = "Arrhenius": s(u) = A * e^(- B / u)
        B : float
            - For mode = "logistic": s(u) = A * u^2 - B * u
            - For mode = "Arrhenius": s(u) = A * e^(- B / u)
    
        Returns
        -------
        _None_
        """
        self._check_configuration(mode=mode, amp=amp, delta=delta, period=period, A=A, B=B)

        self.sigma = sigma
        self.center = center
        self.mode = mode
        self.amp = amp
        self.delta = delta
        self.period = period
        self.A = A
        self.B = B
        self.fn = self._get_source_fn()

    def _check_configuration(
            self,
            mode: str,
            amp: float,
            delta: float,
            period: float,
            A: float,
            B: float
    ) -> None:
        """
        Check the acceptability of a configuration.
        """
        if mode in ["constant", "decay", "periodic", "temporary"]:
            self._is_implicit = False
        elif mode in ["logistic", "AllenCahn", "Arrhenius"]:
            self._is_implicit = True
        else:
            raise ValueError(f"Argument 'mode' must be 'constant'|'decay'|'periodic'|'temporary'|'logistic'|'AllenCahn'|'Arrhenius', not {mode}.")
        
        if not self._is_implicit:
            if amp is None:
                raise ValueError(f"Explicit source function requires to specify the amp parameter (amplitude).")
            if mode == "decay" and delta is None:
                raise ValueError(f"Decaying source function requires to specify the delta parameter (decay factor).")
            if mode == "periodic" and period is None:
                raise ValueError(f"Periodic source function requires to specify the period parameter (signal period).")
            if mode == "temporary" and period is None:
                raise ValueError(f"Temporary source function requires to specify the period parameter (signal duration).")
        else:
            if A is None:
                raise ValueError(f"Implicit source function requires to specify the A parameter.")
            if B is None and mode != "AllenCahn":
                raise ValueError(f"Implicit source function requires to specify the B parameter.")

    def _get_source_fn(self) -> Callable:
        """
        Returns the source function corresponding to the state of the object.
        """
        if self.mode == "constant":
            # Constant source
            def source_fn(x, y, **kwargs):
                if x is None or y is None:
                    raise ValueError(f"A constant source requires spatial coordinates: x = {x}, y = {y}.")
                return self.amp * self._G(x, y) 
                    
        elif self.mode == "decay":
            def source_fn(x, y, t, **kwargs):
                # Decaying source
                if x is None or y is None or t is None:
                    raise ValueError(f"A decaying source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")
                return self.amp * np.exp(- self.delta * t) * self._G(x, y)
                    
        elif self.mode == "periodic":
            # Periodic source
            def source_fn(x, y, t, **kwargs):
                if x is None or y is None or t is None:
                    raise ValueError(f"A periodic source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")       
                w = 2 * np.pi / self.period
                return self.amp * np.sin(w * t) * self._G(x, y)
                
        elif self.mode == "temporary":
            # Temporary source
            def source_fn(x, y, t, **kwargs):
                if x is None or y is None or t is None:
                    raise ValueError(f"A temporary source requires spatio-temporal coordinates: x = {x}, y = {y}, t = {t}.")
                return self.amp * self._G(x, y) * (t < self.period)
            
        elif self.mode == "logistic":
            # Logistic source
            def source_fn(u, x = None, y = None, t = None):    
                if u is None:
                    raise TypeError(f"A logistic source requires 'u'.")
                return self.A * u ** 2 - self.B * u
                
        elif self.mode == "AllenCahn":
            # Allen-Cahn-type source
            def source_fn(u, **kwargs):
                if u is None:
                    raise TypeError(f"An AllenCahn source requires 'u'.")
                return self.A * (u ** 3 - u)
                        
        elif self.mode == "Arrhenius":
            # Arrhenius-type source
            def source_fn(u, **kwargs):
                if u is None:
                    raise TypeError(f"An Arrhenius source requires 'u'.")
                return self.A * np.exp(- self.B / u)

        return source_fn

    def _G(# Gaussian spot
            self,
            x: np.ndarray | torch.Tensor, 
            y: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Gaussian bump at (x, y).
        """
        xc, yc = self.center
        return np.exp(- ((x - xc) ** 2 + (y - yc) ** 2)/(2 * self.sigma ** 2))

    def set_sigma(self, sigma: float) -> None:
        self.sigma = sigma
        self.fn = self._get_source_fn()

    def set_center(self, center: Tuple[float, float]) -> None:
        self.center = center
        self.fn = self._get_source_fn()

    def set_amp(self, amp: float) -> None:
        self._check_configuration(mode=self.mode, amp=amp, delta=self.delta, period=self.period, A=self.A, B=self.B)
        self.amp = amp
        self.fn = self._get_source_fn()

    def set_delta(self, delta: float) -> None:
        self._check_configuration(mode=self.mode, amp=self.amp, delta=delta, period=self.period, A=self.A, B=self.B)
        self.delta = delta
        self.fn = self._get_source_fn()

    def set_delta(self, delta: float) -> None:
        self._check_configuration(mode=self.mode, amp=self.amp, delta=delta, period=self.period, A=self.A, B=self.B)
        self.delta = delta
        self.fn = self._get_source_fn()

    def set_period(self, period: float) -> None:
        self._check_configuration(mode=self.mode, amp=self.amp, delta=self.delta, period=period, A=self.A, B=self.B)
        self.period = period
        self.fn = self._get_source_fn()

    def set_A(self, A: float) -> None:
        self._check_configuration(mode=self.mode, amp=self.amp, delta=self.delta, period=self.period, A=A, B=self.B)
        self.A = A
        self.fn = self._get_source_fn()

    def set_B(self, B: float) -> None:
        self._check_configuration(mode=self.mode, amp=self.amp, delta=self.delta, period=self.period, A=self.A, B=B)
        self.B = B
        self.fn = self._get_source_fn()

    def __call__(
            self,
            x: np.ndarray | torch.Tensor = None,
            y: np.ndarray | torch.Tensor = None,
            t: np.ndarray | torch.Tensor = None,
            u: np.ndarray | torch.Tensor = None
    ) -> np.ndarray | torch.Tensor:
        """
        Call funcction.
        """
        return self.fn(x=x, y=y, t=t, u=u)

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the object.
        """
        return {
            "mode": self.mode,
            "amp": self.amp,
            "center": self.center,
            "sigma": self.sigma,
            "delta": self.delta,
            "period": self.period,
            "A": self.A,
            "B": self.B
        }

    def load_state(self, state: dict):
        """
        Loads the given state into the object.
        """
        self._check_configuration(
            mode=state["mode"], 
            amp=state["amp"], 
            delta=state["delta"], 
            period=state["period"], 
            A=state["A"], 
            B=state["B"]
        )
        self.mode = state["mode"]
        self.amp = state["amp"]
        self.center = state["center"]
        self.sigma = state["sigma"]
        self.delta = state["delta"]
        self.period = state["period"]
        self.A = state["A"]
        self.B = state["B"]
        self.fn = self._get_source_fn()

    def mode_view(self) -> dict:
        """
        Returns a dictionary of the object state taking part in the call process (the interesting pieces for the current mode).
        """
        if self.mode == "constant":
            return {
                "mode": self.mode, 
                "amp": self.amp
            }
        
        elif self.mode == "decay":
            return {
                "mode": self.mode, 
                "amp": self.amp,
                "center": self.center,
                "sigma": self.sigma,
                "delta": self.delta
            }
        elif self.mode == "periodic" or self.mode == "temporary":
            return {
                "mode": self.mode, 
                "amp": self.amp,
                "center": self.center,
                "sigma": self.sigma,
                "period": self.period
            }
                    
        elif self.mode == "logistic" or self.mode == "Arrhenius":
            return {
                "mode": self.mode, 
                "A": self.A,
                "B": self.B
            }
                        
        elif self.mode == "AllenCahn":
            return {
                "mode": self.mode, 
                "A": self.A
            }

    @classmethod
    def null_source(cls) -> Self:
        return Source(
            mode="constant",
            center=(0.0, 0.0),
            amp=0.0
        )