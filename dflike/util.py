def emulators_to_jax(parser, desired=["tt", "te", "ee"]):
    """
        Convert the Cl emulators found in a Cosmopower.YAMLParser into
        ComsmoPower_JAX emulators.
    """
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

    emulators = parser.restore_networks()

    results = {}

    for xy in desired:
        if xy == "bb":
            emu = CosmoPowerJAX("cmb_ee")
        else:
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
            emu.param_train_mean = emulators[f"Cl/{xy}"].parameters_mean.numpy()  # noqa: E501
            emu.param_train_std = emulators[f"Cl/{xy}"].parameters_std.numpy()
            emu.feature_train_mean = emulators[f"Cl/{xy}"].features_mean.numpy()  # noqa: E501
            emu.feature_train_std = emulators[f"Cl/{xy}"].features_std.numpy()

        results[xy] = emu

    return results
