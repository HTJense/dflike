import numpy as np
import jax.numpy as jnp

from .cmb_like import MultiFrequency


class Moped(MultiFrequency):
    def __init__(self, config: str | dict):
        super().__init__(config)

        self.moped_B = jnp.array(np.loadtxt(self.config["moped_file"]))
        self.covariance = jnp.eye(self.moped_B.shape[1])

    def get_whitened_residual(self, theta, **kwargs):
        mu = self.get_model(theta, **kwargs)
        return self.moped_B.T @ self.data_vec - mu

    def get_model(self, theta, **kwargs):
        model = super().get_model(theta, **kwargs)
        return self.moped_B.T @ model


def get_cobaya_class():
    from .cobaya import Moped_cobaya

    return Moped_cobaya
