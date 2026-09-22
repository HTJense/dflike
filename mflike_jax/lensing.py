import numpy as np
import sacc
import os
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path


class Lensing_jax:
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
            data_path = os.path.join(resolve_packages_path(), "data", self.config["data_folder"])
        data = sacc.Sacc.load_fits(os.path.join(data_path, self.config["data_file"]))
        fiducial = sacc.Sacc.load_fits(os.path.join(data_path, self.config["fiducial_file"]))
        corrections = sacc.Sacc.load_fits(os.path.join(data_path, self.config["corrections_file"]))

        print("data", data.get_tracer_combinations())
        print("fiducial", fiducial.get_tracer_combinations())
        print("corrections", corrections.get_tracer_combinations())

        self.parameters = []

    @partial(jax.jit, static_argnums=(0,))
    def chisquare(self):
        return 0.0


def get_cobaya_class():
    """
        This function allow one to import the lensing likelihood into cobaya
        using as `mflike_jax.lensing`.
    """
    from .cobaya import Lensing_jax_cobaya

    return Lensing_jax_cobaya
