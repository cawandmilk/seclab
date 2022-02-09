import tensorflow as tf

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method                ## FGS
from cleverhans.tf2.attacks.madry_et_al import madry_et_al                                  ## PGD
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2                      ## CW
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent    ## BIM
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method      ## MIM

###

import numpy as np

from pathlib import Path
from typing import Callable, Dict


def set_gpu_growthable():
    """ Set gpu memory growthable.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            ## Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
        except RuntimeError as e:
            ## Memory growth must be set before GPUs have been initialized
            print(e)


def get_attack_fn(attack_type: str) -> Callable:
    """ Get attack function for every attack method.
    """
    assert attack_type.upper() in ["FGS", "PGD", "CW", "BIM", "MIM"]
    attack_type = attack_type.upper()
    
    return {
        "FGS": fast_gradient_method,
        "PGD": madry_et_al,
        "CW": carlini_wagner_l2,
        "BIM": projected_gradient_descent,
        "MIM": momentum_iterative_method,
    }[attack_type]


def get_params(attack_type: str) -> dict:
    """ Get hyperparameters for every attack method.
    """
    assert attack_type.upper() in ["FGS", "PGD", "CW", "BIM", "MIM"]
    attack_type = attack_type.upper()

    return {
        "FGS": {
            ## "model_fn": "",
            ## "x": "",
            "eps": 0.3,
            "norm": np.inf,
            "loss_fn": None,
            "clip_min": 0.,
            "clip_max": 1.,
            "y": None,
            "targeted": False,
            "sanity_checks": False,
        },
        "PGD": {
            ## "model_fn": "",
            ## "x": "",
            "eps": 0.3,
            "eps_iter": 0.01,
            "nb_iter": 40,
            "norm": np.inf,
            "clip_min": 0.,
            "clip_max": 1.,
            "y": None,
            "targeted": False,
            "rand_minmax": 0.3,
            "sanity_checks": False,
        },
        "CW": {
            ## "model_fn": "",
            "y": None,
            "targeted": False,
            "batch_size": 1,
            "clip_min": 0.,
            "clip_max": 1.,
            "binary_search_steps": 5,
            "max_iterations": 1_000,
            "abort_early": True,
            "confidence": 40., ## kappa
            "initial_const": 1e-2,
            "learning_rate": 5e-3,
        },
        "BIM": {
            ## "model_fn": "",
            ## "x": "",
            "eps": 0.3,
            "eps_iter": 0.01,
            "nb_iter": 40,
            "norm": np.inf,
            "loss_fn": None,
            "clip_min": 0.,
            "clip_max": 1.,
            "y": None,
            "targeted": False,
            "rand_init": None,
            "rand_minmax": None,
            "sanity_checks": False,
        },
        "MIM": {
            ## "model_fn": "",
            ## "x": "",
            "eps": 0.3,
            "eps_iter": 0.01,
            "nb_iter": 40,
            "norm": np.inf,
            "clip_min": 0.,
            "clip_max": 1.,
            "y": None,
            "targeted": False,
            "decay_factor": 1.,
            "sanity_checks": True,
        },
    }[attack_type]


def load_data(parent: str, suffix: str = "data") -> Dict[str, np.ndarray]:
    """ Load every numpy data.
    """
    segments = {"inp": [], "tar": []}

    print(parent)

    for element in Path(parent).glob(f"*.{suffix}"):
        inp, tar = np.load(element, allow_pickle=True)
        segments["inp"].append(inp)
        segments["tar"].append(tar)

    segments["inp"] = np.concatenate(segments["inp"], axis=0)
    segments["tar"] = np.concatenate(segments["tar"], axis=0)

    return segments


def load_model(parent: str, suffix: str = "h5", without_softmax: bool = True) -> tf.keras.Model:
    """ Load pretrained model.
    """
    model = tf.keras.models.load_model(str(list(Path(parent).glob(f"*.{suffix}"))[0]))
    if without_softmax:
        model.layers[-1].activation = None

    return model
