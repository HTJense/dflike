import jax
import jax.numpy as jnp
import numpy as np


def newton_raphson(func, x0, n_steps, param_ranges=None, alpha0=1.0,
                   tqdm=lambda x, **kwargs: x):
    """
        Boundary-aware Newton-Raphson optimizer.
        Finds the maximum of func(x) by iterating over
            x_(i+1) = x_i - H(x_i)^-1 * ∇f(x_i)

        Optionally can use hard parameter bounds x ∈ [a,b] through the mapping
            x(z) = a + (b - a) * σ(z)
        and optimizing over z.
    """
    if param_ranges is not None:
        lower = jnp.array([x[0] for x in param_ranges])
        upper = jnp.array([x[1] for x in param_ranges])
        x0 = jnp.clip(x0, lower + 1e-4, upper - 1e-4)

        z_to_x = jax.jit(lambda z: lower + (upper - lower) * jax.nn.sigmoid(z))
        f = jax.jit(lambda z: func(z_to_x(z)))
        z = -jnp.log((upper - x0) / (x0 - lower))
    else:
        def z_to_x(z): return z
        f = func
        z = x0

    grad = jax.jit(jax.grad(f))
    hess = jax.jit(jax.hessian(f))

    chain = [x0]
    eps = 1e-4
    c = 1e-4

    for i in tqdm(range(n_steps), leave=True):
        H = hess(z)
        g = grad(z)

        dz = jnp.linalg.solve(H + eps * jnp.eye(len(g)), -g)

        if jnp.any(jnp.isnan(dz)):
            return chain

        alpha = alpha0

        # Check if we are overshooting the minimum.
        while f(z + alpha * dz) > f(z) + c * alpha * g @ dz:
            alpha *= 0.5

        z = z + alpha * dz
        chain.append(z_to_x(z))

        # Step size very small, we're close to convergence.
        if jnp.linalg.norm(dz, ord=jnp.inf) < 1e-5:
            return chain

    return chain


def newton_raphson_multi(func, x0, n_steps, param_ranges=None, alpha0=1.0,
                         tqdm=lambda x, **kwargs: x):
    """
        Multiple Newton-Raphson minimizers from a distribution of points x0.
    """
    if param_ranges is not None:
        lower = jnp.array([x[0] for x in param_ranges])
        upper = jnp.array([x[1] for x in param_ranges])
        x0 = [jnp.clip(x, lower + 1e-4, upper - 1e-4) for x in x0]

        z_to_x = jax.jit(lambda z: lower + (upper - lower) * jax.nn.sigmoid(z))
        f = jax.jit(lambda z: func(z_to_x(z)))
    else:
        def z_to_x(z): return z
        f = func
        z = x0

    grad = jax.jit(jax.grad(f))
    hess = jax.jit(jax.hessian(f))

    chains = []

    for x in tqdm(x0, leave=False):
        chain = [x]
        z = -jnp.log((upper - x) / (x - lower))
        eps = 1e-6
        c = 1e-4

        for i in tqdm(range(n_steps), leave=False):
            H = hess(z)
            g = grad(z)

            dz = jnp.linalg.solve(H + eps * jnp.eye(len(g)), -g)

            if jnp.any(jnp.isnan(dz)):
                break

            alpha = alpha0

            while f(z + alpha * dz) > f(z) + c * alpha * g @ dz:
                alpha *= 0.5

            z = z + alpha * dz
            chain.append(z_to_x(z))

            if jnp.linalg.norm(dz, ord=jnp.inf) < 1e-5:
                break

        chains.append(np.array(chain))

    return chains


def emulators_to_jax(parser, desired=["tt", "te", "ee"]):
    """
        Convert the Cl emulators found in a Cosmopower.YAMLParser into
        ComsmoPower_JAX emulators.
    """
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

    emulators = parser.restore_networks()

    results = {}

    for xy in desired:
        emu = CosmoPowerJAX(f"cmb_{xy}")
        emu.n_parameters = emulators[f"Cl/{xy}"].n_parameters
        emu.parameters = [str(x) for x in emulators[f"Cl/{xy}"].parameters]
        emu.modes = emulators[f"Cl/{xy}"].modes

        weights = [w.numpy().T for w in emulators[f"Cl/{xy}"].W]
        biases = [b.numpy() for b in emulators[f"Cl/{xy}"].b]
        alphas = [a.numpy() for a in emulators[f"Cl/{xy}"].alphas]
        betas = [b.numpy() for b in emulators[f"Cl/{xy}"].betas]

        emu.weights = list(zip(weights, biases))
        emu.hyper_params = list(zip(alphas, betas))

        if hasattr(emulators[f"Cl/{xy}"], "n_pcas"):
            # PCA data.
            emu.n_pcas = emulators[f"Cl/{xy}"].n_pcas
            emu.pca_matrix = emulators[f"Cl/{xy}"].pca_transform_matrix_
            emu.param_train_mean = emulators[f"Cl/{xy}"].parameters_mean_
            emu.param_train_std = emulators[f"Cl/{xy}"].parameters_std_
            emu.feature_train_mean = emulators[f"Cl/{xy}"].pca_mean_
            emu.feature_train_std = emulators[f"Cl/{xy}"].pca_std_
            emu.training_mean = emulators[f"Cl/{xy}"].features_mean_
            emu.training_std = emulators[f"Cl/{xy}"].features_std_
        else:
            # Non-PCA data.
            emu.param_train_mean = emulators[f"Cl/{xy}"].parameters_mean.numpy()
            emu.param_train_std = emulators[f"Cl/{xy}"].parameters_std.numpy()
            emu.feature_train_mean = emulators[f"Cl/{xy}"].features_mean.numpy()
            emu.feature_train_std = emulators[f"Cl/{xy}"].features_std.numpy()

        results[xy] = emu

    return results
