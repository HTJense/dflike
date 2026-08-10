from .likelihood import MFLike_jax
from .bandpower_foregrounds import BandpowerForegrounds

def get_cobaya_class():
	from .cobaya import MFLike_jax_cobaya

	return MFLike_jax_cobaya
