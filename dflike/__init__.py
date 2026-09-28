from .bandpower_foregrounds import BandpowerForegrounds  # noqa: F401
from .cosmopower import Cosmopower  # noqa: F401
from .lensing import Lensing_jax  # noqa: F401
from .lensing_corrections import LensingCorrections  # noqa: F401
from .cmb_like import MultiFrequency, CMBLite  # noqa: F401
from .pipeline import Pipeline  # noqa: F401
from .prior import Prior  # noqa: F401


def get_cobaya_class():
    from .cobaya import MultiFrequency_cobaya

    return MultiFrequency_cobaya
