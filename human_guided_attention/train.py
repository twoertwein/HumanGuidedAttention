#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from python_tools import caching, generic
from python_tools.generic import namespace_as_string
from python_tools.ml import metrics, neural
from python_tools.ml.default.neural_models import AttenuatedModalityExperts
from python_tools.ml.default.transformations import (DefaultTransformations,
                                                     revert_transform,
                                                     set_transform)
from python_tools.ml.evaluator import evaluator

from human_guided_attention.model import Attenuated_Modality_Experts_Human


def get_modalities(columns: list[str]) -> dict[str, list[str]]:
    vision = [x for x in columns if x.startswith("openface_") or x.startswith("afar")]
    liwc = [x for x in columns if x.startswith("liwc")]
    acoustic = [
        x
        for x in columns
        if generic.startswith_list(
            x, ("opensmile_", "volume_", "covarep_"), reverse=True
        )
    ]
    return {"vision": vision, "acoustic": acoustic, "liwc": liwc}


def combine_transformations(data, transform, model_transform=None):
    data = set_transform(data, transform)
    data.add_transform(model_transform, optimizable=True)
    return data


def get_training_name(args):
    return namespace_as_string(args, exclude=("WORKERS",))


def train(args):
    partitions, Y_names = caching.read_pickle(Path("data.pickle"))
    name = f"Guided={args.GUIDED}"

    folder = Path(name)
    print(folder)

    """
    # add your own features
    # there are 5 family-independent folds
    for fold in partitions.values():
        # each with a training, valdiation and test set
        for dataloader in fold.values():
            # the dataloader has a list of all the data (.iterator)
            # and a dictionary with meta data (features names and so on,
            # .property_dict)
            dataloader.property_dict["X_names"] = np.concatenate(
                [dataloader.property_dict["X_names"], np.array(["new_feature"])]
            )
            for batch in dataloader.iterator:
                # to align your features, you will need to use the following fields:
                # batch["meta_id"][0], family id (negative if it is the mother)
                # batch["meta_start"][0], start of the segment in seconds
                # batch["meta_end"][0], end of the segment in seconds

                # add new feature
                batch["X"][0] = np.concatenate(
                    [batch["X"][0], np.ones((batch["X"][0].shape[0], 1))], axis=1
                )
    """

    # init default modules
    params = {
        "nominal": True,
        "metric_max": True,
        "y_names": np.asarray(Y_names),
    }
    metric_fun = partial(metrics.nominal_metrics, names=Y_names)
    metric = "accuracy_balanced"

    # get model+parameter and losses
    neural_kwargs = {
        "epochs": [5000],
        "early_stop": [100],
        "lr": [1e-06, 1e-05, 0.0001, 0.001, 0.005, 0.01],
        "attenuation": [""],
        "final_activation": [{"name": "linear"}],
        "dropout": [0.0],
        "activation": [{"name": "ReLU"}],
        "layers": [0, 1, 2],
        "weight_decay": [0.0, 1e-4, 1e-2],
        "sample_weight": [True],
    }
    if args.GUIDED:
        neural_kwargs["attenuation_lambda"] = [0.1, 0.5, 1.0]

    # wrap pool in attenuated experts
    model = AttenuatedModalityExperts(device="cuda", **params)
    for key in list(neural_kwargs.keys()):
        if not (key in ("dropout", "activation", "layers") or key.startswith("model_")):
            continue
        neural_kwargs[f"model_{key}"] = [(x,) for x in neural_kwargs.pop(key)]
    neural_kwargs["model"] = [neural.MLP]
    feature_names = partitions[0]["training"].property_dict["X_names"].tolist()
    columns = get_modalities(feature_names)
    neural_kwargs["input_sizes"] = [
        [tuple(columns["vision"]), tuple(columns["acoustic"]), tuple(columns["liwc"])]
    ]
    neural_kwargs["competitive"] = [False]
    neural_kwargs["joint_attenuation"] = [
        None,
        {
            "final_activation": {"name": "linear"},
            "activation": {"name": "ReLU"},
            "dropout": 0.0,
            "layers": 0,
        },
    ]
    neural_kwargs["latent_gating"] = [True]
    neural_kwargs["combinations"] = [((0,), (1,), (2,))]

    for key, value in neural_kwargs.items():
        if not key.startswith("model_") or isinstance(value[0], tuple):
            continue
        neural_kwargs[key] = [(x,) for x in value]
    neural_kwargs["model_class"] = [Attenuated_Modality_Experts_Human]
    neural_kwargs["guided"] = [args.GUIDED]
    model.forward_names = tuple(list(model.forward_names) + ["guided"])

    model.update_parameter(neural_kwargs)

    for key, value in model.parameters.items():
        if len(value) == 1:
            continue
        print(key, len(value), str(value)[: min(len(str(value)), 500)])

    models, parameters, transform_ = model.get_models()

    # transformations
    transform = DefaultTransformations(**params)
    transform_parameter = ({"feature_selection_svm": True},) * len(partitions)

    # combine overall and model-specific transform
    apply_transformation = partial(combine_transformations, model_transform=transform_)
    models = (models[0],) * len(parameters)

    # parallel options
    kwargs = {"parallel": "local", "n_workers": args.WORKERS}

    print("number of models", len(models))
    evaluator(
        models=models,
        partitions=partitions,
        workers=args.WORKERS,
        parameters=parameters,
        folder=folder,
        metric_fun=metric_fun,
        metric=metric,
        metric_max=params["metric_max"],
        learn_transform=transform.define_transform,
        apply_transform=apply_transformation,
        revert_transform=revert_transform,
        transform_parameter=transform_parameter,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--WORKERS", type=int, default=1)
    parser.add_argument("--GUIDED", action="store_const", const=True, default=False)
    args = parser.parse_args()

    train(args)
