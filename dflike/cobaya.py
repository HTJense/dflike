"""
    Cobaya wrappers for DFLike.
"""
from . import cmb_like
from . import bandpower_foregrounds as fg
from . import lensing, lensing_corrections
from cobaya.likelihood import Likelihood
import numpy as np
from typing import Optional


class MultiFrequency_cobaya(Likelihood):
    like_config_file: Optional[str | dict] = None
    fg_config_file: Optional[str | dict] = None

    def initialize(self):
        self.like = cmb_like.MultiFrequency(self.like_config_file)
        self.theory = fg.BandpowerForegrounds(self.fg_config_file, self.like)

    def get_requirements(self):
        reqs = {
            "Cl": {k: self.like.lmax for k in self.like.requested_cls}
        }

        for par in self.like.parameters + self.theory.parameters:
            reqs[par] = None

        return reqs

    def logp(self, **params):
        cls = self.provider.get_Cl(ell_factor=True)
        theta_fg = np.array([params[k] for k in self.theory.parameters])
        foregrounds = self.theory.get_foreground_model(theta_fg)
        theta_like = np.array([params[k] for k in self.like.parameters])
        chi2 = self.like.chisquare(theta_like, cls=cls,
                                   foregrounds=foregrounds)

        self.log.debug(f"Chi square = {chi2:.2f}")

        return float(-chi2 / 2.)


class Lensing_cobaya(Likelihood):
    config_file: str | dict
    corr_config_file: Optional[str | dict] = None

    def initialize(self):
        self.like = lensing.Lensing(self.config_file)
        if self.corr_config_file is not None:
            self.theory = lensing_corrections.LensingCorrections(
                self.corr_config_file
            )
        else:
            self.theory = None

    def get_requirements(self):
        reqs = {
            "Cl": {"pp": self.like.lmax}
        }

        for par in self.like.parameters:
            reqs[par] = None

        if self.theory is not None:
            reqs["Cl"] |= {
                "tt": self.theory.lmax,
                "te": self.theory.lmax,
                "ee": self.theory.lmax,
                "bb": self.theory.lmax
            }

            for par in self.theory.parameters:
                reqs[par] = None

        return reqs

    def logp(self, **params):
        cls = self.provider.get_Cl(ell_factor=True)
        corr = None
        if self.theory is not None:
            theta_th = np.array([params[k] for k in self.theory.parameters])
            corr = self.theory.get_corrections(theta_th, cls=cls)

        theta_like = np.array([params[k] for k in self.like.parameters])
        chi2 = self.like.chisquare(theta_like, cls=cls, corrections=corr)

        self.log.debug(f"Chi square = {chi2:.2f}")

        return float(-chi2 / 2.)
