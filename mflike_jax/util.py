import jax
import jax.numpy as jnp
import numpy as np
import warnings


def newton_raphson(func, x0, n_steps, param_ranges=None, alpha0=1.0, tqdm=lambda x, **kwargs: x):
	"""
		Boundary-aware Newton-Raphson optimizer.
		Finds the maximum of func(x) by iterating over
			x_(i+1) = x_i - H(x_i)^-1 * ∇f(x_i)
		
		Optionally can use hard parameter bounds x ∈ [a,b] through the mapping
			x(z) = a + (b - a) * σ(z)
		and optimizing over z.
	"""
	if param_ranges is not None:
		lower = jnp.array([ x[0] for x in param_ranges ])
		upper = jnp.array([ x[1] for x in param_ranges ])
		x0 = jnp.clip(x0, lower + 1e-4, upper - 1e-4)

		z_to_x = jax.jit(lambda z: lower + (upper - lower) * jax.nn.sigmoid(z))
		f = jax.jit(lambda z: func(z_to_x(z)))
		z = -jnp.log((upper - x0) / (x0 - lower))
	else:
		z_to_x = lambda z: z
		f = func
		z = x0

	grad = jax.jit(jax.grad(f))
	hess = jax.jit(jax.hessian(f))

	chain = [x0]
	eps = 1e-6
	c = 1e-4

	for i in (tbar := tqdm(range(n_steps), leave=True)):
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
		if jnp.linalg.norm(dz, ord=jnp.inf) < 1e-8:
			return chain

	return chain


def newton_raphson_multi(func, x0, n_steps, param_ranges=None, alpha0=1.0, tqdm=lambda x, **kwargs: x):
	"""
		Multiple Newton-Raphson minimizers from a distribution of points x0.
	"""
	if param_ranges is not None:
		lower = jnp.array([ x[0] for x in param_ranges ])
		upper = jnp.array([ x[1] for x in param_ranges ])
		x0 = [ jnp.clip(x, lower + 1e-4, upper - 1e-4) for x in x0 ]

		z_to_x = jax.jit(lambda z: lower + (upper - lower) * jax.nn.sigmoid(z))
		f = jax.jit(lambda z: func(z_to_x(z)))
	else:
		z_to_x = lambda z: z
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

		for i in (tbar := tqdm(range(n_steps), leave=False)):
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

			if jnp.linalg.norm(dz, ord=jnp.inf) < 1e-8:
				break

		chains.append(np.array(chain))
	
	return chains


def emulators_to_jax(parser):
	"""
		Convert the Cl emulators found in a Cosmopower.YAMLParser into ComsmoPower_JAX emulators.
	"""
	from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
	
	emulators = parser.restore_networks()
	
	# I apologize for this code,
	# But the high-accuracy emulators created for cosmopower are not compatible with cosmopower-jax
	# So we have to manually convert them into the right format.......
	# TT
	cmb_tt = CosmoPowerJAX("cmb_tt")
	cmb_tt.n_parameters = emulators["Cl/tt"].n_parameters
	cmb_tt.parameters = [str(x) for x in emulators["Cl/tt"].parameters]
	cmb_tt.param_train_mean = emulators["Cl/tt"].parameters_mean.numpy()
	cmb_tt.param_train_std = emulators["Cl/tt"].parameters_std.numpy()
	cmb_tt.feature_train_mean = emulators["Cl/tt"].features_mean.numpy()
	cmb_tt.feature_train_std = emulators["Cl/tt"].features_std.numpy()
	cmb_tt.modes = emulators["Cl/tt"].modes

	weights = [ w.numpy().T for w in emulators["Cl/tt"].W ]
	biases = [ b.numpy() for b in emulators["Cl/tt"].b ]
	alphas = [ a.numpy() for a in emulators["Cl/tt"].alphas ]
	betas = [ b.numpy() for b in emulators["Cl/tt"].betas ]

	cmb_tt.weights = list(zip(weights, biases))
	cmb_tt.hyper_params = list(zip(alphas, betas))

	# TE
	cmb_te = CosmoPowerJAX("cmb_te")
	cmb_te.n_parameters = emulators["Cl/te"].n_parameters
	cmb_te.parameters = [str(x) for x in emulators["Cl/te"].parameters]
	cmb_te.n_pcas = emulators["Cl/te"].n_pcas
	cmb_te.pca_matrix = emulators["Cl/te"].pca_transform_matrix_
	cmb_te.param_train_mean = emulators["Cl/te"].parameters_mean_
	cmb_te.param_train_std = emulators["Cl/te"].parameters_std_
	cmb_te.feature_train_mean = emulators["Cl/te"].pca_mean_
	cmb_te.feature_train_std = emulators["Cl/te"].pca_std_
	cmb_te.training_mean = emulators["Cl/te"].features_mean_
	cmb_te.training_std = emulators["Cl/te"].features_std_
	cmb_te.modes = emulators["Cl/te"].modes

	weights = [ w.numpy().T for w in emulators["Cl/te"].W ]
	biases = [ b.numpy() for b in emulators["Cl/te"].b ]
	alphas = [ a.numpy() for a in emulators["Cl/te"].alphas ]
	betas = [ b.numpy() for b in emulators["Cl/te"].betas ]

	cmb_te.weights = list(zip(weights, biases))
	cmb_te.hyper_params = list(zip(alphas, betas))

	# EE
	cmb_ee = CosmoPowerJAX("cmb_ee")
	cmb_ee.n_parameters = emulators["Cl/ee"].n_parameters
	cmb_ee.parameters = [str(x) for x in emulators["Cl/ee"].parameters]
	cmb_ee.param_train_mean = emulators["Cl/ee"].parameters_mean.numpy()
	cmb_ee.param_train_std = emulators["Cl/ee"].parameters_std.numpy()
	cmb_ee.feature_train_mean = emulators["Cl/ee"].features_mean.numpy()
	cmb_ee.feature_train_std = emulators["Cl/ee"].features_std.numpy()
	cmb_ee.modes = emulators["Cl/ee"].modes

	weights = [ w.numpy().T for w in emulators["Cl/ee"].W ]
	biases = [ b.numpy() for b in emulators["Cl/ee"].b ]
	alphas = [ a.numpy() for a in emulators["Cl/ee"].alphas ]
	betas = [ b.numpy() for b in emulators["Cl/ee"].betas ]

	cmb_ee.weights = list(zip(weights, biases))
	cmb_ee.hyper_params = list(zip(alphas, betas))

	return cmb_tt, cmb_te, cmb_ee
