import numpy as np
import sacc
import os
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path


class MFLike_jax:
    def __init__(self, config):
        self.config = yaml_load_file(config)

        data_path = os.path.join(resolve_packages_path(), "data", self.config["data_folder"])
        s = sacc.Sacc.load_fits(os.path.join(data_path, self.config["input_file"]))

        defaults = self.config["defaults"]

        self.experiments = self.config["data"]["experiments"]
        self.tracers = {}
        for name, tracer in s.tracers.items():
            self.tracers[name] = {"nu": tracer.nu, "bp": tracer.bandpass, "beam": tracer.beam.T}

        data_vec = []
        data_indices = []
        self.spec_meta = []
        self.ells = jnp.arange(2, defaults["lmax"]+1)
        i0 = 0

        for entry in self.config["data"]["spectra"]:
            ex1, ex2 = entry["experiments"]
            for xy in entry.get("polarizations", defaults["polarizations"]):
                dt = {"TT": "cl_00", "TE": "cl_0e", "ET": "cl_0e", "EE": "cl_ee"}[xy]
                spec = {"TT": 0, "TE": 1, "ET": 1, "EE": 2}[xy]
                t1, t2 = ex1 + ("_s0" if xy[0] == "T" else "_s2"), ex2 + ("_s0" if xy[1] == "T" else "_s2")
                lmin, lmax = entry["scales"][xy]

                ell, cl, ind = s.get_ell_cl(dt, t1, t2, return_ind=True)
                m = np.logical_and(ell > lmin, ell < lmax)
                data_indices += list(ind[m])
                data_vec += list(cl[m])
                bpw = s.get_bandpower_windows(ind)

                self.spec_meta.append({
                    "dt": dt,
                    "exp1": ex1,
                    "exp2": ex2,
                    "ix": (self.experiments.index(ex1), self.experiments.index(ex2)),
                    "lmin": lmin,
                    "lmax": lmax,
                    "ids": i0 + np.arange(len(ind[m])),
                    "leff": ell[m],
                    "spec": spec,
                    "window": jnp.array(bpw.weight.T[m]),
                    "values": jnp.array(bpw.values)
                })
                i0 += len(ind[m])

        self.data_vec = np.array(data_vec)
        data_indices = np.array(data_indices)

        self.covmat = s.covariance.covmat[data_indices,:][:,data_indices]
        self.inv_cov = np.linalg.inv(self.covmat)

    @partial(jax.jit, static_argnums=(0,))
    def bin_spectra(self, spec):
        model_vec = jnp.zeros_like(self.data_vec)
        for block in self.spec_meta:
            ids = block["ids"]
            xy = block["spec"]
            ls = block["values"] - 2
            ix, jx = block["ix"]

            sub = spec[xy,ls,ix,jx]

            model_vec = model_vec.at[ids].set(block["window"] @ sub)

        return model_vec

    @partial(jax.jit, static_argnums=(0,))
    def calibrate_spectra(self, spectra, **params):
        """ spectra is a (3, ell, exp1, exp2) array """
        """ for each pair of (exp1, exp2), we divide the entry by """
        """ (cal_exp1 * cal_exp2, cal_exp1 * cal_exp2 * calE_exp2, cal_exp1 * calE_exp1 * cal_exp2 * calE_exp2 """
        calib_spec = jnp.zeros_like(spectra)
        calG = 1. / params["calG_all"] ** 2.

        for i, exp1 in enumerate(self.experiments):
            for j, exp2 in enumerate(self.experiments):
                calib_spec = calib_spec.at[0,:,i,j].set( spectra[0,:,i,j] * calG / (params[f"cal_{exp1}"] * params[f"cal_{exp2}"]) )
                calib_spec = calib_spec.at[1,:,i,j].set( spectra[1,:,i,j] * calG / (params[f"cal_{exp1}"] * params[f"cal_{exp2}"] * params[f"calE_{exp2}"]) )
                calib_spec = calib_spec.at[2,:,i,j].set( spectra[2,:,i,j] * calG / (params[f"cal_{exp1}"] * params[f"calE_{exp1}"] * params[f"cal_{exp2}"] * params[f"calE_{exp2}"]) )

        return calib_spec

    @partial(jax.jit, static_argnums=(0,))
    def get_unbinned_model(self, dltt, dlte, dlee, foregrounds, **params):
        """Project (TT,TE,EE) from (3, ell) to (3, ell, exp1, exp2) and add foregrounds."""
        spec = jnp.stack([ dltt[self.ells], dlte[self.ells], dlee[self.ells] ])
        foregrounds = foregrounds[:,self.ells,:,:]
        spec = jnp.broadcast_to(spec[:,:,None,None], foregrounds.shape) + foregrounds

        return self.calibrate_spectra(spec, **params)

    @partial(jax.jit, static_argnums=(0,))
    def chisquare(self, dltt, dlte, dlee, foregrounds, **params):
        spec = self.get_unbinned_model(dltt, dlte, dlee, foregrounds, **params)
        binned_spec = self.bin_spectra(spec)
        delta = self.data_vec - binned_spec
        chi2 = delta @ self.inv_cov @ delta
        return chi2
