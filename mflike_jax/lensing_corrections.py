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
            data_path = os.path.join(resolve_packages_path(), "data", self.config["data_folder"])

        fiducial = sacc.Sacc.load_fits(os.path.join(data_path, self.config["fiducial_file"]))
        corrections = sacc.Sacc.load_fits(os.path.join(data_path, self.config["corrections_file"]))

    def get_corrections(self, theta):
        return 0.0
