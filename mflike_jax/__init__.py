from .likelihood import MFLike_jax  # noqa: F401
from .bandpower_foregrounds import BandpowerForegrounds  # noqa: F401
from .lensing import Lensing_jax  # noqa: F401
from .lensing_corrections import LensingCorrections  # noqa: F401


def get_cobaya_class():
    from .cobaya import MFLike_jax_cobaya

    return MFLike_jax_cobaya
