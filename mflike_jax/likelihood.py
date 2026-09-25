import numpy as np
import sacc
import os
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path


NORMALIZE_SPECTRA = {
    "tt": "tt",
    "te": "te",
    "tb": "tb",
    "et": "te",
    "ee": "ee",
    "eb": "eb",
    "bt": "tb",
    "be": "eb",
    "bb": "bb"
}

MFLIKE_DATA_TYPES = {
    "tt": "cl_00",
    "te": "cl_0e",
    "et": "cl_0e",
    "tb": "cl_0b",
    "bt": "cl_0b",
    "ee": "cl_ee",
    "eb": "cl_eb",
    "be": "cl_eb",
    "bb": "cl_bb"
}


class MFLike_jax:
    def __init__(self, config: str | dict):
        if type(config) is str:
            self.config = yaml_load_file(config)
        elif type(config) is dict:
            self.config = config
        else:
            raise TypeError("Configuration should be string (filename) or "
                            f"dictionary, but was given {type(config)}.")

        data_path = self.config["data_folder"]
        if not os.path.isdir(data_path):
            data_path = os.path.join(resolve_packages_path(), "data",
                                     self.config["data_folder"])
        s = sacc.Sacc.load_fits(os.path.join(data_path,
                                             self.config["input_file"]))
        
        if self.config.get("cov_Bbl_file", None) is None:
            t = s
        else:
            t = sacc.Sacc.load_fits(os.path.join(data_path,
                                                 self.config["cov_Bbl_file"]))

        self.requested_cls = [
            xy.lower() for xy in self.config["requested_cls"]
        ]
        defaults = self.config["defaults"]

        self.experiments = self.config["data"]["experiments"]
        self.tracers = {}
        for name, tracer in s.tracers.items():
            self.tracers[name] = {
                "nu": tracer.nu,
                "bp": tracer.bandpass,
                "beam": tracer.beam.T
            }

        data_vec = []
        data_indices = []
        self.lmax = defaults["lmax"]
        self.spec_meta = []
        self.ells = jnp.arange(2, self.lmax+1)
        i0 = 0

        self.parameters = ["calG_all"]
        for exp in self.experiments:
            self.parameters.append(f"cal_{exp}")
            self.parameters.append(f"calE_{exp}")

        self.calT_index = jnp.array([
            self.parameters.index(f"cal_{exp}") for exp in self.experiments
        ])
        self.calE_index = jnp.array([
            self.parameters.index(f"calE_{exp}") for exp in self.experiments
        ])

        for entry in self.config["data"]["spectra"]:
            ex1, ex2 = entry["experiments"]
            for xy in map(lambda x: x.lower(), entry.get("polarizations", defaults["polarizations"])):
                dt = MFLIKE_DATA_TYPES[xy]
                spec = self.requested_cls.index(NORMALIZE_SPECTRA[xy])

                t1 = ex1 + ("_s0" if xy[0] == "t" else "_s2")
                t2 = ex2 + ("_s0" if xy[1] == "t" else "_s2")
                lmin, lmax = entry["scales"][xy.upper()]

                ell, cl, ind = s.get_ell_cl(dt, t1, t2, return_ind=True)
                m = np.logical_and(ell > lmin, ell < lmax)
                data_indices += list(ind[m])
                data_vec += list(cl[m])
                bpw = t.get_bandpower_windows(ind)

                x1, x2 = (
                    (ex2, ex1) if xy in ["et", "bt", "be"] else (ex1, ex2)
                )

                self.spec_meta.append({
                    "dt": dt,
                    "exp1": x1,
                    "exp2": x2,
                    "ix": (self.experiments.index(x1),
                           self.experiments.index(x2)),
                    "lmin": lmin,
                    "lmax": lmax,
                    "ids": i0 + np.arange(len(ind[m])),
                    "leff": ell[m],
                    "spec": spec,
                    "window": jnp.array(bpw.weight.T[m]),
                    "values": jnp.array(bpw.values)
                })
                i0 += len(ind[m])

        self.data_vec = jnp.asarray(data_vec)
        data_indices = np.array(data_indices)

        self.covmat = t.covariance.covmat[data_indices, :][:, data_indices]
        self.inv_cov = jnp.asarray(np.linalg.inv(self.covmat))
        self.logp_const = -0.5 * (
            np.log(2. * np.pi) * len(self.data_vec)
            + np.linalg.slogdet(self.inv_cov)[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def bin_spectra(self, spec):
        model_vec = jnp.zeros_like(self.data_vec)
        for block in self.spec_meta:
            ids = block["ids"]
            xy = block["spec"]
            ls = block["values"] - 2
            ix, jx = block["ix"]

            sub = spec[xy, ls, ix, jx]

            model_vec = model_vec.at[ids].set(block["window"] @ sub)

        return model_vec

    @partial(jax.jit, static_argnums=(0,))
    def calibrate_spectra(self, spectra, theta):
        """ spectra is a (xy, ell, exp1, exp2) array
            for each pair of (exp1, exp2), we divide the entry by the
            respective calibration factor. """
        calG = 1. / theta[0] ** 2.
        calT = 1. / theta[self.calT_index]
        calE = 1. / (theta[self.calT_index] * theta[self.calE_index])

        res = []
        for i, xy in enumerate(self.requested_cls):
            cal = calG
            if xy[0] == "t":
                cal *= calT[:,None]
            else:
                cal *= calE[:,None]
            if xy[1] == "t":
                cal *= calT[None,:]
            else:
                cal *= calE[None,:]

            res.append(spectra[i] * cal)

        return jnp.stack(res)

    @partial(jax.jit, static_argnums=(0,))
    def get_unbinned_model(self, cls, foregrounds, theta):
        """ Project (cls) from (xy, ell) to (xy, ell, exp1, exp2) and add
            foregrounds."""
        spec = jnp.stack([
            cls[xy][self.ells] for xy in self.requested_cls
        ])
        spec = (jnp.broadcast_to(spec[:, :, None, None], foregrounds.shape)
                + foregrounds)

        return self.calibrate_spectra(spec, theta)

    @partial(jax.jit, static_argnums=(0,))
    def get_model(self, cls, foregrounds, theta):
        """ Get the binned data model. """
        model = self.get_unbinned_model(cls, foregrounds, theta)
        return self.bin_spectra(model)

    @partial(jax.jit, static_argnums=(0,))
    def chisquare(self, cls, foregrounds, theta):
        model = self.get_model(cls, foregrounds, theta)
        delta = self.data_vec - model
        chi2 = delta @ self.inv_cov @ delta
        return chi2

    @partial(jax.jit, static_argnums=(0,))
    def loglike(self, cls, foregrounds, theta):
        chi2 = self.chisquare(cls, foregrounds, theta)
        return -0.5 * chi2 + self.logp_const

    @partial(jax.jit, static_argnums=(0, 2))
    def design_matrix(self, theta, n_ig=0):
        nb = {}
        # Count the number of bins per extracted cross-spectrum.
        for m in self.spec_meta:
            dt = m["dt"]
            nb[dt] = max(nb.get(dt, 0), len(m["ids"]) - n_ig)
        b = 0
        b0 = {}
        # Find the zero-point offset per spectrum.
        for dt in nb:
            b0[dt] = b
            b += nb[dt]
        M = jnp.zeros((len(self.data_vec), sum(nb.values())))

        # Start filling out the design matrix.
        for m in self.spec_meta:
            dt = m["dt"]
            x1, x2 = m["exp1"], m["exp2"]
            ids = m["ids"]
            calG = theta[0]
            calT1 = theta[self.parameters.index(f"cal_{x1}")]
            calT2 = theta[self.parameters.index(f"cal_{x2}")]
            calE1 = theta[self.parameters.index(f"calE_{x1}")]
            calE2 = theta[self.parameters.index(f"calE_{x2}")]
            cal = 1. / (calG ** 2. * calT1 * calT2)
            if dt in ["cal_0e", "cal_0b"]:
                cal /= calE2
            if dt in ["cal_ee", "cal_eb", "cal_bb"]:
                cal /= (calE1 * calE2)

            b = b0[dt] + np.arange(nb[dt] - len(ids) + n_ig, nb[dt])
            M = M.at[ids[:-n_ig], b].set(cal)

        return M
