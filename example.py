import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from likelihood import MFLike_jax
from bandpower_foregrounds import BandpowerForegrounds
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

params = fg_params | sys_params

free_params = like.parameters + fg_model.parameters

priors = {
    "calG_all": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.003),

    "cal_dr6_pa4_f220": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.013),
    "cal_dr6_pa5_f090": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.0016),
    "cal_dr6_pa5_f150": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.0020),
    "cal_dr6_pa6_f090": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.0018),
    "cal_dr6_pa6_f150": lambda x: jsc.stats.norm.logpdf(x, loc=1.0, scale=0.0024),

    "bandint_shift_dr6_pa4_f220": lambda x: jsc.stats.norm.logpdf(x, loc=0.0, scale=3.6),
    "bandint_shift_dr6_pa5_f090": lambda x: jsc.stats.norm.logpdf(x, loc=0.0, scale=1.0),
    "bandint_shift_dr6_pa5_f150": lambda x: jsc.stats.norm.logpdf(x, loc=0.0, scale=1.3),
    "bandint_shift_dr6_pa6_f090": lambda x: jsc.stats.norm.logpdf(x, loc=0.0, scale=1.2),
    "bandint_shift_dr6_pa6_f150": lambda x: jsc.stats.norm.logpdf(x, loc=0.0, scale=1.1),

    "a_gtt": lambda x: jsc.stats.norm.logpdf(x, loc=7.95, scale=0.32),
    "a_gte": lambda x: jsc.stats.norm.logpdf(x, loc=0.432, scale=0.03),
    "a_gee": lambda x: jsc.stats.norm.logpdf(x, loc=0.1681, scale=0.017),
}

""" JAX likes working with vectors rather than dictionaries, so we are going to
    turn our dictionary into a vector (and then turn it back later, don't
    think too hard about it). """

theta = jnp.array([ params[k] for k in free_params ])

fixed_params = {
    k : params[k] for k in params if k not in free_params
}

""" Define our log-likelihood and log-prior functions. Note that these are
    *negative* log(X), because some standard implementations are searching for
    function minima rather than maxima. """

@jax.jit
def logprior(theta):
    params = { k: theta[i] for i, k in enumerate(free_params) } | fixed_params
    logp = 0.0
    for k in priors:
        logp = logp + priors[k](params[k])
    return -logp

@jax.jit
def loglike(theta):
    theta_like = theta[:len(like.parameters)]
    theta_fg = theta[len(like.parameters):]
    foregrounds = fg_model.get_foreground_model(theta_fg)
    logl = -0.5 * like.chisquare(spec[:,0], spec[:,3], spec[:,1], foregrounds, theta_like)
    return -logl

@jax.jit
def logpost(theta):
    return logprior(theta) + loglike(theta)


""" We can simply evaluate these functions directly: """
print("log(prior) = ", logprior(theta))
print("log(like)  = ", loglike(theta))
print("log(post)  = ", logpost(theta))

""" Or we can do some jax things with them: """
grad = jax.grad(logpost)
hess = jax.hessian(logpost)

# These two are functions, we can evaluate them at the same values and find
# derivatives:
H = hess(theta)
cov = jnp.linalg.inv(H)
err = np.sqrt(np.diag(cov))

for i, par in enumerate(free_params):
    print(f"{par}: {theta[i]:.2f} +/- {err[i]:.2f}")

