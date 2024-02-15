def get_and_check_prior_choice(FLAGS):
    """Create a dictionary of choices of hierarchical priors.
    Also check whether there are no contradiction in the choices.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    prior_choice = {
        "theta": FLAGS.theta,
        "exp_verbosity": FLAGS.exp_verbosity,
        "beta": FLAGS.beta,
        "eta": FLAGS.eta,
        "ideal_dim": FLAGS.ideal_dim,
        "ideal_mean": FLAGS.ideal_mean,
        "ideal_prec": FLAGS.ideal_prec,
        "iota_dim": FLAGS.iota_dim,
        "iota_mean": FLAGS.iota_mean,
        "iota_prec": FLAGS.iota_prec,
    }

    if prior_choice['theta'] == 'Gfix' and prior_choice['exp_verbosity'] == 'None':
        raise Warning('The model does not adjust for any verbosity at all. '
                      'Either make more hierarchical prior for theta (Garte, Gdrte) or '
                      'assume some prior for exp_verbosity, e.g. Gfix (CAVI) LNfix (SG).')
    if prior_choice['theta'] in ['Garte', 'Gdrte'] and prior_choice['exp_verbosity'] in ['Gfix', 'LNfix']:
        raise ValueError('The model cannot contain two parametrizations of verbosity at once.'
                         'Either set prior choice for theta to Gfix and keep exp_verbosity or '
                         'remove exp_verbosity from the model (None) and keep hierarchical prior for theta.')

    if prior_choice['ideal_dim'] == 'a' and prior_choice['iota_dim'] == 'kl':
        raise ValueError('Cannot make topic-specific regression coefficients without topic-specific ideological positions.'
                         'Either make ideological positions topic-specific (ideal_dim = ak) or '
                         'make regression coefficients fixed for all topics (iota_dim = l).')

    if prior_choice['iota_mean'] == 'Nlmean' and prior_choice['ideal_mean'] != 'Nreg':
        raise ValueError('Since there is no regression assumed, iotas and iota_means should not exist.'
                         'Either assume some regression by ideal_mean = Nreg or '
                         'set iota_mean = None to make the mean fixed.')

    if prior_choice['iota_prec'] in ["NlprecG", "NlprecF"] and prior_choice['ideal_mean'] != 'Nreg':
        raise ValueError('Since there is no regression assumed, iotas and iota_means should not exist.'
                         'Either assume some regression by ideal_mean = Nreg or '
                         'set iota_prec = Nfix to make the precision fixed.')

    return prior_choice

def get_and_check_prior_hyperparameter(FLAGS):
    """Create a dictionary of hyperparameters for hierarchical priors.
    Also check whether their values fit the necessary conditions.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    if FLAGS.eta == "NkprecG":
        eta_prec_rte = FLAGS.eta_prec_rte * 2.0 / FLAGS.eta_kappa
    else:
        eta_prec_rte = FLAGS.eta_prec_rte

    if FLAGS.eta == "NkprecF":
        # switch the shapes
        eta_prec_shp = FLAGS.eta_prec_rate_shp
        eta_prec_rate_shp = FLAGS.eta_prec_shp
        # adjust rates by kappa
        eta_prec_rate_rte = FLAGS.eta_prec_shp / FLAGS.eta_prec_rate_shp * FLAGS.eta_kappa / 2.0
    else:
        eta_prec_shp = FLAGS.eta_prec_shp
        eta_prec_rate_shp = FLAGS.eta_prec_rate_shp
        eta_prec_rate_rte = FLAGS.eta_prec_rate_rte

    # Hyperparameters for triple gamma prior for iota adjustments:
    if FLAGS.iota_prec == "NlprecG":
        iota_prec_rte = FLAGS.iota_prec_rte * 2.0 / FLAGS.iota_kappa
    else:
        iota_prec_rte = FLAGS.iota_prec_rte

    if FLAGS.iota_prec == "NlprecF":
        # switch the shapes
        iota_prec_shp = FLAGS.iota_prec_rate_shp
        iota_prec_rate_shp = FLAGS.iota_prec_shp
        # adjust rates by kappa
        iota_prec_rate_rte = FLAGS.iota_prec_shp / FLAGS.iota_prec_rate_shp * FLAGS.iota_kappa / 2.0
    else:
        iota_prec_shp = FLAGS.iota_prec_shp
        iota_prec_rate_shp = FLAGS.iota_prec_rate_shp
        iota_prec_rate_rte = FLAGS.iota_prec_rate_rte

    prior_hyperparameter = {
        "theta": {"shape": FLAGS.theta_shp, "rate": FLAGS.theta_rte},
        "theta_rate": {"shape": FLAGS.theta_rate_shp, "rate": FLAGS.theta_rate_shp / FLAGS.theta_rate_mean},
        "beta": {"shape": FLAGS.beta_shp, "rate": FLAGS.beta_rte},
        "beta_rate": {"shape": FLAGS.beta_rate_shp, "rate": FLAGS.beta_rate_shp / FLAGS.beta_rate_mean},
        "exp_verbosity": {"location": FLAGS.exp_verbosity_loc, "scale": FLAGS.exp_verbosity_scl,
                          "shape": FLAGS.exp_verbosity_shp, "rate": FLAGS.exp_verbosity_rte},
        "eta": {"location": FLAGS.eta_loc, "scale": FLAGS.eta_scl},
        "eta_prec": {"shape": eta_prec_shp, "rate": eta_prec_rte},
        "eta_prec_rate": {"shape": eta_prec_rate_shp, "rate": eta_prec_rate_rte},
        "ideal": {"location": FLAGS.ideal_loc, "scale": FLAGS.ideal_scl},
        "ideal_prec": {"shape": FLAGS.ideal_prec_shp, "rate": FLAGS.ideal_prec_rte},
        "iota": {"location": FLAGS.iota_loc, "scale": FLAGS.iota_scl},
        "iota_prec": {"shape": iota_prec_shp, "rate": iota_prec_rte},
        "iota_prec_rate": {"shape": iota_prec_rate_shp, "rate": iota_prec_rate_rte},
        "iota_mean": {"location": FLAGS.iota_mean_loc, "scale": FLAGS.iota_mean_scl},
    }

    # theta
    if prior_hyperparameter['theta']['shape'] <= 0:
        raise ValueError('Hyperparameter theta:shp is not positive.')
    if prior_hyperparameter['theta']['rate'] <= 0:
        raise ValueError('Hyperparameter theta:rte is not positive.')
    if prior_hyperparameter['theta_rate']['shape'] <= 0:
        raise ValueError('Hyperparameter theta_rate:shp is not positive.')
    if prior_hyperparameter['theta_rate']['rate'] <= 0:
        raise ValueError('Hyperparameter theta_rate:rte is not positive.')

    # beta
    if prior_hyperparameter['beta']['shape'] <= 0:
        raise ValueError('Hyperparameter beta:shp is not positive.')
    if prior_hyperparameter['beta']['rate'] <= 0:
        raise ValueError('Hyperparameter beta:rte is not positive.')
    if prior_hyperparameter['beta_rate']['shape'] <= 0:
        raise ValueError('Hyperparameter beta_rate:shp is not positive.')
    if prior_hyperparameter['beta_rate']['rate'] <= 0:
        raise ValueError('Hyperparameter beta_rate:rte is not positive.')

    # exp_verbosity
    if prior_hyperparameter['exp_verbosity']['shape'] <= 0:
        raise ValueError('Hyperparameter exp_verbosity:shp is not positive.')
    if prior_hyperparameter['exp_verbosity']['rate'] <= 0:
        raise ValueError('Hyperparameter exp_verbosity:rte is not positive.')

    # eta
    if prior_hyperparameter['eta']['scale'] <= 0:
        raise ValueError('Hyperparameter eta:scl is not positive.')
    if prior_hyperparameter['eta_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter eta_prec:shp is not positive.')
    if prior_hyperparameter['eta_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter eta_prec:rte is not positive.')
    if prior_hyperparameter['eta_prec_rate']['shape'] <= 0:
        raise ValueError('Hyperparameter eta_prec_rate:shp is not positive.')
    if prior_hyperparameter['eta_prec_rate']['rate'] <= 0:
        raise ValueError('Hyperparameter eta_prec_rate:rte is not positive.')

    # ideal
    if prior_hyperparameter['ideal']['scale'] <= 0:
        raise ValueError('Hyperparameter ideal:scl is not positive.')
    if prior_hyperparameter['ideal_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter ideal_prec:shp is not positive.')
    if prior_hyperparameter['ideal_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter ideal_prec:rte is not positive.')

    # iota
    if prior_hyperparameter['iota']['scale'] <= 0:
        raise ValueError('Hyperparameter iota:scl is not positive.')
    if prior_hyperparameter['iota_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter iota_prec:shp is not positive.')
    if prior_hyperparameter['iota_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter iota_prec:rte is not positive.')
    if prior_hyperparameter['iota_prec_rate']['shape'] <= 0:
        raise ValueError('Hyperparameter iota_prec_rate:shp is not positive.')
    if prior_hyperparameter['iota_prec_rate']['rate'] <= 0:
        raise ValueError('Hyperparameter iota_prec_rate:rte is not positive.')
    if prior_hyperparameter['iota_mean']['scale'] <= 0:
        raise ValueError('Hyperparameter iota_mean:scl is not positive.')


    return prior_hyperparameter