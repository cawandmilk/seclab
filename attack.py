import tensorflow as tf

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method                ## FGS
from cleverhans.tf2.attacks.madry_et_al import madry_et_al                                  ## PGD
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2                      ## CW
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent    ## BIM
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method      ## MIM

###

import argparse
import pprint
import time

import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Callable, Dict

###

from utils import set_gpu_growthable


def define_argparser():
    """ Define arguments.
    """
    p = argparse.ArgumentParser()

    ## Defaults.
    p.add_argument(
        "--global_batch_size",
        type=int,
        default=128,
        help="Batch size for sum of total replica gpus.",
    )
    p.add_argument(
        "--mnist",
        type=str,
        default="./modelNsamples/mnist",
        help="The path of mnist dataset. Default=%(default)s",
    )
    p.add_argument(
        "--cifar",
        type=str,
        default="./modelNsamples/cifar10",
        help="The path of cifar dataset. Default=%(default)s",
    )

    config = p.parse_args()
    return config


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

    if attack_type == "FGS":
        params = {
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
        }

    elif attack_type == "PGD":
        params = {
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
        }

    elif attack_type == "CW":## L2
        params = {
            ## "model_fn": "",
            "y": None,
            "targeted": False,
            "batch_size": 128,
            "clip_min": 0.,
            "clip_max": 1.,
            "binary_search_steps": 5,
            "max_iterations": 1_000,
            "abort_early": True,
            "confidence": 0., ## kappa
            "initial_const": 1e-2,
            "learning_rate": 5e-3,
        }

    elif attack_type == "BIM":
        params = {
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
        }

    elif attack_type == "MIM":
        params = {
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
        }

    return params


def load_data(parent: str, suffix="data") -> Dict[str, np.ndarray]:
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


def load_model(parent: str, suffix="h5") -> tf.keras.Model:
    """ Load pretrained model.
    """
    return tf.keras.models.load_model(str(list(Path(parent).glob(f"*.{suffix}"))[0]))


def make_dataset(config, data: Dict[str, np.ndarray], AUTO=tf.data.AUTOTUNE) -> tf.data.Dataset:

    def _normalize(element):
        element["inp"] = tf.cast(element["inp"], dtype=tf.float32) / 255.
        element["tar"] = tf.cast(element["tar"], dtype=tf.int32)
        return element

    return (
        tf.data.Dataset.from_tensor_slices(data)
        .map(_normalize, num_parallel_calls=AUTO)
        .cache()
        .batch(config.global_batch_size, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )


def attack(ds: tf.data.Dataset, model_fn: tf.keras.Model, attack_fn: Callable, attack_type: str, params: dict) -> np.ndarray:
    """ Do attack and evaluate it.
    """

    def _eval(_x: tf.Tensor, _y: tf.Tensor) -> float:
        return tf.math.reduce_mean(
            tf.keras.metrics.sparse_categorical_accuracy(
                y_true=_y, 
                y_pred=model_fn.predict(_x))
        ).numpy()

    xs = []
    ys = []
    xs_adv = []

    start = time.time()

    for element in tqdm(ds):
        ## Unpack.
        x, y = element["inp"], element["tar"]

        ## Generate adversarial examples.
        x_adv = attack_fn(
            model_fn=model_fn,
            x=x,
            **params,
        )
        
        ## Gather.
        xs.append(x)
        ys.append(y)
        xs_adv.append(x_adv)

    end = time.time()

    ## Concat.
    xs = tf.concat(xs, axis=0)
    ys = tf.concat(ys, axis=0)
    xs_adv = tf.concat(xs_adv, axis=0)

    ## Box constraint for valid images.
    xs_adv = tf.cast(xs_adv * 255, tf.uint8)
    xs_adv = tf.cast(xs_adv, tf.float32) / 255.

    ## Calculate metrics.
    acc = _eval(xs, ys)
    acc_adv = _eval(xs_adv, ys)

    ## Print.
    print(f"[{attack_type.upper()}] acc: {acc * 100:.1f}, acc (adv): {acc_adv * 100:.1f}, exec time: {end - start:.6f}")

    ## Return adversarial examples.
    return tf.cast(xs_adv * 255, tf.uint8).numpy()


def main(config):
    """ Main body.
    """
    def print_config(config):
        ## 'sort_dicts=False' params can only apply python>=3.8.
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)
    print_config(vars(config))

    ## Set gpu memory growthable.
    set_gpu_growthable()

    ## Load data.
    mnist_npy = load_data(config.mnist)
    cifar_npy = load_data(config.cifar)

    print("[MNIST] " + ", ".join([f"{key}.shape: {value.shape}" for key, value in mnist_npy.items()]))
    print("[CIFAR] " + ", ".join([f"{key}.shape: {value.shape}" for key, value in cifar_npy.items()]))

    ## Make dataset.
    mnist_ds = make_dataset(config, data=mnist_npy)
    cifar_ds = make_dataset(config, data=cifar_npy)

    print(f"[MNIST] {mnist_ds}")
    print(f"[CIFAR] {cifar_ds}")

    ## Load (pretrained) model.
    model_mnist = load_model(config.mnist) ## LeNet5
    model_cifar = load_model(config.cifar) ## VGG19

    ## Do attack.
    for ds_type, ds in {"mnist": mnist_ds, "cifar": cifar_ds}.items():
        ## We only use LeNet5 when dataset is mnist, else VGG19. (not grid search)
        model_fn = model_mnist if ds_type == "mnist" else model_cifar

        ## For every attack types...
        for attack_type in ["FGS", "PGD", "BIM", "MIM"]: ## CW
            x_adv = attack(
                ds=ds,
                model_fn=model_fn,
                attack_fn=get_attack_fn(attack_type),
                attack_type=attack_type,
                params=get_params(attack_type),
            )

            ## Save it.
            save_to = config.mnist if ds_type == "mnist" else config.cifar

            np.save(Path(save_to, f"{attack_type.upper()}_1000"), x_adv[:1000])
            np.save(Path(save_to, f"{attack_type.upper()}_next_1000"), x_adv[1000:])

    ## EOF.


if __name__ == "__main__":
    config = define_argparser()
    main(config)
