import tensorflow as tf

###

import argparse
import json
import pprint
import time

import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Callable, Dict

###

from utils import set_gpu_growthable, get_attack_fn, get_params, load_data, load_model


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


def make_dataset(config, data: Dict[str, np.ndarray], batch_size: int = None, AUTO=tf.data.AUTOTUNE) -> tf.data.Dataset:
    """ Generate dataset.
    """
    def _normalize(element):
        element["inp"] = tf.cast(element["inp"], dtype=tf.float32) / 255.
        element["tar"] = tf.cast(element["tar"], dtype=tf.int32)
        return element

    return (
        tf.data.Dataset.from_tensor_slices(data)
        .map(_normalize, num_parallel_calls=AUTO)
        .cache()
        .batch(batch_size if batch_size != None else config.global_batch_size, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )


def attack(ds: tf.data.Dataset, model_fn: tf.keras.Model, attack_fn: Callable, params: dict, desc: str) -> np.ndarray:
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

    for element in tqdm(ds, desc=desc):
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
    # xs_adv = tf.cast(xs_adv * 255, tf.uint8)
    # xs_adv = tf.cast(xs_adv, tf.float32) / 255.

    ## Calculate metrics.
    acc = _eval(xs, ys)
    acc_adv = _eval(xs_adv, ys)

    ## Results.
    return (
        tf.cast(xs_adv * 255, tf.uint8).numpy(), 
        {
            "acc": acc * 100,
            "acc_adv": acc_adv * 100,
            "exec_time": end - start,
        }
    )


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

    mnist_ds_cw = make_dataset(config, data=mnist_npy, batch_size=1)
    cifar_ds_cw = make_dataset(config, data=cifar_npy, batch_size=1)

    print(f"[MNIST] {mnist_ds}")
    print(f"[CIFAR] {cifar_ds}")

    ## Load (pretrained) model.
    model_mnist = load_model(config.mnist) ## LeNet5
    model_cifar = load_model(config.cifar) ## VGG19

    ## Do attack.
    results = []
    for ds_type in ["mnist", "cifar"]:
        ## We only use LeNet5 when dataset is mnist, else VGG19. (not grid search)
        model_fn = model_mnist if ds_type == "mnist" else model_cifar

        ## For every attack types...
        for attack_type in ["FGS", "PGD", "CW", "BIM", "MIM"]: ## CW
            ## CW attack allow only batch_size=1.
            if attack_type == "CW":
                ds = mnist_ds_cw if ds_type == "mnist" else cifar_ds_cw
            else:
                ds = mnist_ds if ds_type == "mnist" else cifar_ds

            x_adv, result = attack(
                ds=ds,
                model_fn=model_fn,
                attack_fn=get_attack_fn(attack_type),
                params=get_params(attack_type),
                desc=f"{ds_type.upper()}-{attack_type:<3}",
            )

            result["ds_type"] = ds_type
            result["save_to"] = config.mnist if ds_type == "mnist" else config.cifar
            result["attack_type"] = attack_type

            ## Save it.
            np.save(Path(result["save_to"], f"{result['attack_type'].upper()}_1000"), x_adv[:1000])
            np.save(Path(result["save_to"], f"{result['attack_type'].upper()}_next_1000"), x_adv[1000:])

            ## Append it.
            results.append(result)

    ## Print and save results.
    print_config(results)

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
