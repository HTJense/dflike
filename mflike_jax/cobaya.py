"""
    Cobaya wrappers for mflike_jax.
"""
from . import likelihood as like
from . import bandpower_foregrounds as fg
from . import lensing, lensing_corrections
from cobaya.likelihood import Likelihood
import numpy as np
from typing import Optional


class MFLike_jax_cobaya(Likelihood):
    like_config_file: Optional[str | dict] = None
    fg_config_file: Optional[str | dict] = None

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
        theta_fg = np.array([params[k] for k in self.theory.parameters])
        fg_model = self.theory.get_foreground_model(theta_fg)
        theta_like = np.array([params[k] for k in self.like.parameters])
        chi2 = self.like.chisquare(cl["tt"], cl["te"], cl["ee"], fg_model,
                                   theta_like)

        self.log.debug(f"Chi square = {chi2:.2f}")

        return float(-chi2 / 2.)


class Lensing_jax_cobaya(Likelihood):
    config_file: Optional[str | dict] = None
    corr_config_file: Optional[str | dict] = None

    def initialize(self):
        self.like = lensing.Lensing_jax(self.config_file)
        self.theory = lensing_corrections.LensingCorrections(
            self.corr_config_file
        )

    def get_requirements(self):
        reqs = {}

        for par in self.like.parameters + self.theory.parameters:
            reqs[par] = None

        return reqs

    def logp(self, **params):
        cl = self.provider.get_Cl(ell_factor=True)
<<<<<<< HEAD

        corr = None
        if self.theory is not None:
            theta_th = np.array([params[k] for k in self.theory.parameters])
            corr = self.theory.get_corrections(theta_th)
=======
        theta_th = np.array([params[k] for k in self.theory.parameters])
        corr = self.theory.get_corrections(theta_th)
>>>>>>> parent of 64211d2 (Made lensing corrections optional.)

        theta_like = np.array([params[k] for k in self.like.parameters])
        chi2 = self.like.chisquare(cl["pp"], corr, theta_like)

        self.log.debug(f"Chi square = {chi2:.2f}")

        return float(-chi2 / 2.)
