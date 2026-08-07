import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from mflike_jax import MFLike_jax, BandpowerForegrounds
import camb

""" One could use Cosmopower-JAX here for cosmology, but I'll keep cosmology
    fixed for this example. """

cosmo_params = {
    "ombh2": 0.0225,
    "omch2": 0.1193,
    "cosmomc_theta": 104.073e-4,
    "ns": 0.9709,
    "logA": 3.056,
    "tau": 0.0603
}

cosmo_params["As"] = 1e-10 * np.exp(cosmo_params.pop("logA"))

pars = camb.set_params(lens_potential_accuracy=8, lmax=8500, **cosmo_params)
res = camb.get_results(pars)
spec = res.get_cmb_power_spectra(CMB_unit="muK")["total"]

""" We can initialize the likelihood and foreground model with the exact same
    MFLike yaml files as the official MFLike likelihood. """
like = MFLike_jax("act_dr6.yaml")
fg_model = BandpowerForegrounds("act_dr6_foregrounds.yaml", like)

""" These parameters are (approximately) the ACT best-fitting foreground params
    from Louis et al (2025). """

fg_params = {
    "a_tSZ": 3.4,
    "alpha_tSZ": -0.5,
    "a_kSZ": 1.5,
    "a_c": 3.7,
    "a_p": 7.7,
    "beta_p": 1.9,
    "beta_c": 1.9,
    "xi": 0.09,
    "a_s": 2.9,
    "beta_s": -2.8,
    "a_pste": -0.025,
    "a_psee": 0.02,

    "a_gtt": 7.95,
    "a_gte": 0.423,
    "a_gee": 0.1681,

    "alpha_dT": -0.6,
    "alpha_dE": -0.4,
    "alpha_s": 1.0,
    "T_effd": 19.6,
    "T_d": 9.7,
    "beta_d": 1.5,
    "alpha_p": 1.0,
    "alpha_c": 0.8,
}

sys_params = {
    "calG_all": 1.0,

    "cal_dr6_pa4_f220": 0.98,
    "cal_dr6_pa5_f090": 1.0,
    "cal_dr6_pa5_f150": 1.0,
    "cal_dr6_pa6_f090": 1.0,
    "cal_dr6_pa6_f150": 1.0,

    "calE_dr6_pa4_f220": 1.0,
    "calE_dr6_pa5_f090": 0.985,
    "calE_dr6_pa5_f150": 0.995,
    "calE_dr6_pa6_f090": 1.00,
    "calE_dr6_pa6_f150": 0.995,

    "bandint_shift_dr6_pa4_f220": 5.0,
    "bandint_shift_dr6_pa5_f090": 0.0,
    "bandint_shift_dr6_pa5_f150": -1.0,
    "bandint_shift_dr6_pa6_f090": 1.0,
    "bandint_shift_dr6_pa6_f150": 0.0,
}

params_values = fg_params | sys_params

model_params = like.parameters + fg_model.parameters

priors = {
    "calG_all": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.003),

    "cal_dr6_pa4_f220": lambda x: jsc.stats.norm.logpdf(x, loc=1.0,
                                                        scale=0.013),
    "cal_dr6_pa5_f090": lambda x: jsc.stats.norm.logpdf(x, loc=1.0,
                                                        scale=0.0016),
    "cal_dr6_pa5_f150": lambda x: jsc.stats.norm.logpdf(x, loc=1.0,
                                                        scale=0.0020),
    "cal_dr6_pa6_f090": lambda x: jsc.stats.norm.logpdf(x, loc=1.0,
                                                        scale=0.0018),
    "cal_dr6_pa6_f150": lambda x: jsc.stats.norm.logpdf(x, loc=1.0,
                                                        scale=0.0024),

    "bandint_shift_dr6_pa4_f220": lambda x: jsc.stats.norm.logpdf(x, loc=0.0,
                                                                  scale=3.6),
    "bandint_shift_dr6_pa5_f090": lambda x: jsc.stats.norm.logpdf(x, loc=0.0,
                                                                  scale=1.0),
    "bandint_shift_dr6_pa5_f150": lambda x: jsc.stats.norm.logpdf(x, loc=0.0,
                                                                  scale=1.3),
    "bandint_shift_dr6_pa6_f090": lambda x: jsc.stats.norm.logpdf(x, loc=0.0,
                                                                  scale=1.2),
    "bandint_shift_dr6_pa6_f150": lambda x: jsc.stats.norm.logpdf(x, loc=0.0,
                                                                  scale=1.1),

    "a_gtt": lambda x: jsc.stats.norm.logpdf(x, loc=7.95, scale=0.32),
    "a_gte": lambda x: jsc.stats.norm.logpdf(x, loc=0.432, scale=0.03),
    "a_gee": lambda x: jsc.stats.norm.logpdf(x, loc=0.1681, scale=0.017),
}

""" Define our log-likelihood and log-prior functions. """


@jax.jit
def logprior(theta):
    # Note that this function is not normalized.
    params = {k: theta[i] for i, k in enumerate(model_params)}
    logp = 0.0
    for k in priors:
        logp = logp + priors[k](params[k])
    return logp


@jax.jit
def loglike(theta):
    theta_like = theta[:len(like.parameters)]
    theta_fg = theta[len(like.parameters):]
    foregrounds = fg_model.get_foreground_model(theta_fg)
    logl = -0.5 * like.chisquare(spec[:, 0], spec[:, 3], spec[:, 1],
                                 foregrounds, theta_like)
    return logl


@jax.jit
def logpost(theta):
    return logprior(theta) + loglike(theta)


# These are the parameters with respect to which we want to differentiate.
parameters_baseline = [
    "a_tSZ", "a_kSZ", "alpha_tSZ", "xi",
    "a_c", "beta_c", "a_p", "beta_p", "a_s", "beta_s",
    "a_gtt", "a_gte", "a_gee", "a_pste", "a_psee",
    "calG_all",
    "cal_dr6_pa4_f220", "cal_dr6_pa5_f090", "cal_dr6_pa5_f150",
    "cal_dr6_pa6_f090", "cal_dr6_pa6_f150",
    "calE_dr6_pa5_f090", "calE_dr6_pa5_f150",
    "calE_dr6_pa6_f090", "calE_dr6_pa6_f150",
    "bandint_shift_dr6_pa4_f220", "bandint_shift_dr6_pa5_f090",
    "bandint_shift_dr6_pa5_f150", "bandint_shift_dr6_pa6_f090",
    "bandint_shift_dr6_pa6_f150",
]

# We need a log-posterior function that only takes in baseline parameters and
# keeps the remainder fixed. We create a default list of parameters, and then
# keep track of the indices of the free parameters.
fit_idx = jnp.array([model_params.index(p) for p in parameters_baseline])
theta_default = jnp.array([params_values[p] for p in model_params])


@jax.jit
def logpost_baseline(theta_x):
    # Note that in numpy we could write `theta[fit_idx] = theta_x`
    # But in jax, arrays are immutable, so we need to use this `.at[i].set(x)`.
    theta = theta_default.at[fit_idx].set(theta_x)
    return logpost(theta)


theta_baseline = jnp.array([params_values[p] for p in parameters_baseline])

""" We can simply evaluate these functions directly: """
print(f"log(prior) = {logprior(theta_default):>9.4f}")
print(f"log(like)  = {loglike(theta_default):>9.4f}")
print(f"log(post)  = {logpost(theta_default):>9.4f}")

""" Or we can do some jax things with them: """
grad = jax.grad(logpost_baseline)
hess = jax.hessian(logpost_baseline)

# These two are functions, we can evaluate them at the same values and find
# derivatives:
H = hess(theta_baseline)

# Fisher estimate of the error.
cov = jnp.linalg.inv(-H)
err = np.sqrt(np.diag(cov))

for i, par in enumerate(parameters_baseline):
    print(f"{par:>30s} = {params_values[par]:8.2f} +/- " +
          (f"{err[i]:4.2e}" if err[i] < 0.01 else f"{err[i]:4.2f}"))
