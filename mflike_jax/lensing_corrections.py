import numpy as np
import sacc
import os
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path


class LensingCorrections:
    def __init__(self, config: str | dict):
        if type(config) is str:
            self.config = yaml_load_file(config)
        elif type(config) is dict:
            self.config = config
        else:
            raise TypeError("Configuration should be a string (filename) or "
                            f"dictionary, but was given {type(config)}.")

        data_path = self.config["data_folder"]
        if not os.path.isdir(data_path):
            data_path = os.path.join(resolve_packages_path(), "data",
                                     self.config["data_folder"])

        fiducial = sacc.Sacc.load_fits(
            os.path.join(data_path, self.config["fiducial_file"])
        )
        corrections = sacc.Sacc.load_fits(
            os.path.join(data_path, self.config["corrections_file"])
        )

        self.ells = jnp.array(
            fiducial.get_ell_cl(None, "ct", "ct")[0].astype(int)
        )
        self.lmax = int(self.ells.max())
        self.fiducial = {
            "tt": jnp.array(fiducial.get_ell_cl(None, "ct", "ct")[1]),
            "te": jnp.array(fiducial.get_ell_cl(None, "ct", "ce")[1]),
            "ee": jnp.array(fiducial.get_ell_cl(None, "ce", "ce")[1]),
            "bb": jnp.array(fiducial.get_ell_cl(None, "cb", "cb")[1]),
            "kk": jnp.array(fiducial.get_ell_cl(None, "ck", "ck")[1]),
        }
        self.n0_response = {
            "tt": jnp.array(corrections.get_ell_cl("N0_00", "ct", "ct")[1]),
            "te": jnp.array(corrections.get_ell_cl("N0_0e", "ct", "ce")[1]),
            "ee": jnp.array(corrections.get_ell_cl("N0_ee", "ce", "ce")[1]),
            "bb": jnp.array(corrections.get_ell_cl("N0_bb", "cb", "cb")[1]),
        }
        self.n1_response = {
            "tt": jnp.array(corrections.get_ell_cl("N1_00", "ct", "ct")[1]),
            "te": jnp.array(corrections.get_ell_cl("N1_0e", "ct", "ce")[1]),
            "ee": jnp.array(corrections.get_ell_cl("N1_ee", "ce", "ce")[1]),
            "bb": jnp.array(corrections.get_ell_cl("N1_bb", "cb", "cb")[1]),
        }
        self.n1_clpp = jnp.array(
            corrections.get_ell_cl("N1_00", "cp", "cp")[1]
        )
        self.n0 = jnp.array(corrections.get_ell_cl("N0_00", "n0", "n0")[1][0])

        # The likelihood has C_ell data, so we need some
        # D_ell -> C_ell conversion factors.
        self.dl_factor = jnp.zeros(self.ells.shape)
        self.dl_factor = self.dl_factor.at[self.ells > 0].set(
            (2. * np.pi / (self.ells * (self.ells + 1.)))[self.ells > 0]
        )
        self.kk_factor = 2. * np.pi / 4.
        self.parameters = []

    @partial(jax.jit, static_argnums=(0,))
    def get_corrections(self, cls, theta=None):
        cls = {
            "tt": cls["tt"][self.ells] * self.dl_factor,
            "te": cls["te"][self.ells] * self.dl_factor,
            "ee": cls["ee"][self.ells] * self.dl_factor,
            "bb": cls["bb"][self.ells] * self.dl_factor,
            "kk": cls["pp"][self.ells] * self.kk_factor,
        }
        delta = {s: cls[s] - self.fiducial[s] for s in self.fiducial}
        n0 = sum([self.n0_response[s] @ delta[s] for s in self.n0_response])
        n1 = self.n1_clpp @ (cls["kk"] - self.fiducial["kk"]) \
            + sum([self.n1_response[s] @ delta[s] for s in self.n1_response])

        return 2. * (self.fiducial["kk"] / self.n0) * n0 + n1
