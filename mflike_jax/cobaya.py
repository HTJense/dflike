"""
	Cobaya wrappers for mflike_jax.
"""
from . import likelihood as like
from . import bandpower_foregrounds as fg
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
import numpy as np

class MFLike_jax_cobaya(Likelihood):
	like_config_file: str = None
	fg_config_file: str = None

	def initialize(self):
		self.like = like.MFLike_jax(self.like_config_file)
		self.theory = fg.BandpowerForegrounds(self.fg_config_file, self.like)

	def get_requirements(self):
		reqs = {"Cl": {k: self.like.ells.max() for k in ["tt", "te", "ee"]}}

		for par in self.like.parameters + self.theory.parameters:
			reqs[par] = None
		return reqs

	def logp(self, **params):
		cl = self.provider.get_Cl(ell_factor=True)
		theta_fg = np.array([ params[k] for k in self.theory.parameters ])
		fg_model = self.theory.get_foreground_model(theta_fg)
		theta_like = np.array([ params[k] for k in self.like.parameters ])
		chi2 = self.like.chisquare(cl["tt"], cl["te"], cl["ee"], fg_model, theta_like)

		self.log.debug(f"Chi square = {chi2:.2f}")

		return float(-chi2 / 2.)
