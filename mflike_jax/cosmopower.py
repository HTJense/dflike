from jax import numpy as jnp
from .theory import Theory
from . import util

T_CMB = 2.72548e6


class Cosmopower(Theory):
    def __init__(self, parser):
        self.parameters = set([])
        self.emulators = util.emulators_to_jax(parser)
        for xy in self.emulators:
            self.parameters |= set(self.emulators[xy].parameters)
        self.parameters = list(self.parameters)
        self.indices = {
            xy: jnp.array([
                self.parameters.index(i) for i in self.emulators[xy].parameters
            ])
            for xy in self.emulators
        }

    def compute(self, theta, **kwargs) -> dict:
        cls = {}
        for xy in self.emulators:
            cls[xy] = self.emulators[xy].predict(theta[self.indices[xy]])
            if xy in ["tt", "te", "ee", "bb"]:
                cls[xy] = cls[xy] * (T_CMB ** 2.)

        return {"cls": cls}

    @property
    def inputs(self) -> tuple[str]:
        return ()

    @property
    def outputs(self) -> tuple[str]:
        return ("cls",)
