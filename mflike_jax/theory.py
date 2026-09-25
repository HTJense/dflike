from abc import ABC, abstractmethod


class Theory(ABC):
    @abstractmethod
    def compute(self, theta, **kwargs) -> dict:
        ...

    @property
    @abstractmethod
    def inputs(self) -> tuple[str]:
        ...

    @property
    @abstractmethod
    def outputs(self) -> tuple[str]:
        ...
