import os
import csv
import h5py
import json
import joblib
import hydra
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor

from src.datasets.splitting import CATEGORIES
from src.models import PPO, ResnetProxyModel, ImitationModel


TR = 1.49


def get_activations(
    model,
    data_file,
    data_list,
    layer_list,
    batch_size,
    device,
    out_file_path,
):
    """Apply model to get the activations from the frames in data_file, and save
    the activation in a hdf5 file.

    Args:
        model: model to use
        data_file: hdf5 file with preprocessed frames and bold
        data_list: list of the hdf5 groups to use
        layer_list: list of layers to extract activations from
        batch_size: int, batch size to use
        device: str, torch device to use
        out_file_path: path of the hdf5 file to save the activations
    """

    activations_buffer = {layer_name: [] for layer_name in layer_list}

    ## put hook on model
    def make_hook(name):
        def hook(model, input, output):
            activations_buffer[name].append(output.detach().cpu().clone().numpy())

        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in layer_list:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with h5py.File(out_file_path, "w") as tmp_file:
        for layer_name in layer_list:
            tmp_file.create_group(layer_name)
        for run_name in tqdm(data_list, desc="Computing activations"):
            frames = data_file[run_name]["frames"][:]
            onsets = data_file[run_name]["onsets"][:]
            frames = torch.from_numpy(frames).to(device)
            for i in range(0, len(frames), batch_size):
                with torch.no_grad():
                    _ = model(frames[i : i + batch_size])

            for i, layer_name in enumerate(layer_list):
                activations = np.concatenate(activations_buffer[layer_name], axis=0)
                tmp_file[layer_name].create_group(run_name)
                tmp_file[layer_name][run_name].create_dataset(
                    "activations",
                    data=np.reshape(activations, (activations.shape[0], -1)),
                )
                tmp_file[layer_name][run_name].create_dataset("onsets", data=onsets)
                activations_buffer[layer_name].clear()

    # remove hooks
    for hook in hooks:
        hook.remove()
    del activations_buffer


def get_run_features(run, layer_name, tmp_file_path, data_path, scaler, pca):
    with h5py.File(tmp_file_path, "r") as tmp_file:
        with h5py.File(data_path, "r") as data_file:
            acts = tmp_file[f"{layer_name}/{run}/activations"][:]
            if scaler is not None:
                acts = scaler.transform(acts)
            if pca is not None:
                acts = pca.transform(acts)
            onsets = tmp_file[f"{layer_name}/{run}/onsets"][:]
            durations = np.zeros(len(onsets))
            bold = data_file[f"{run}/bold"][:]
            features = []
            for i in range(acts.shape[1]):
                features.append(
                    compute_regressor(
                        exp_condition=[onsets, durations, acts[:, i]],
                        hrf_model="spm",
                        frame_times=np.arange(len(bold)) * TR + TR / 2,
                    )
                )
    features = np.concatenate([f[0] for f in features], axis=1)
    return features, bold


def get_features_and_targets_prl(
    tmp_file_path, data_path, layer_name, run_list, scaler=None, pca=None, concat=True
):
    features_bold = Parallel(n_jobs=-1)(
        delayed(get_run_features)(
            run_name,
            layer_name,
            tmp_file_path,
            data_path,
            scaler,
            pca,
        )
        for run_name in tqdm(run_list, desc="Generating features")
    )
    features, bold = zip(*features_bold)
    if concat:
        return np.concatenate(features), np.concatenate(bold)
    return features, bold


def get_features_and_targets(
    tmp_file, data_file, layer_name, run_list, scaler=None, pca=None
):
    """Get features and targets for a given layer name, run list, and optional scaling/PCA."""
    all_features = []
    all_bold = []
    for run in tqdm(run_list, desc="Generating features"):
        acts = tmp_file[f"{layer_name}/{run}/activations"][:]
        if scaler is not None:
            acts = scaler.transform(acts)
        if pca is not None:
            acts = pca.transform(acts)
        onsets = tmp_file[f"{layer_name}/{run}/onsets"][:]
        durations = np.zeros(len(onsets))
        bold = data_file[f"{run}/bold"][:]
        features = Parallel(n_jobs=-1)(
            delayed(compute_regressor)(
                exp_condition=[onsets, durations, acts[:, i]],
                hrf_model="spm",
                frame_times=np.arange(len(bold)) * TR + TR / 2,
            )
            for i in range(acts.shape[1])
        )
        features = np.concatenate([f[0] for f in features], axis=1)
        all_features.append(features)
        all_bold.append(bold)
    return np.concatenate(all_features), np.concatenate(all_bold)


@hydra.main(config_path="../conf", config_name="ridge", version_base=None)
def main(cfg):
    """Fit the brain encoding ridge."""
    tmp_dir = cfg.tmp_dir
    if "SLURM_TMPDIR" in os.environ:
        tmp_dir = tmp_dir.replace("$SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if cfg.model == "PPO":
        model = PPO(n_in=4, n_actions=12)
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location=device))
    elif cfg.model == "ResnetModel":
        model = ResnetProxyModel.load_from_checkpoint(
            cfg.ckpt_path, map_location=device
        )
    elif cfg.model == "ImitationModel":
        model = ImitationModel.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    elif cfg.model == "Untrained":
        model = torch.load("models/untrained_model.pt")
    else:
        raise ValueError(f"Invalid model name: {cfg.model}")
    model = model.to(device).eval()
    tng_h5_path = tmp_dir / "train_activations.h5"
    val_h5_path = tmp_dir / "validation_activations.h5"
    test_h5_path = tmp_dir / "test_activations.h5"
    ood_h5_path = tmp_dir / "ood_activations.h5"
    split = json.load(open(cfg.split_path, "r"))
    training_data_list = split["training"]
    validation_data_list = split["validation"]
    test_data_list = split["test"]
    preproc_path = cfg.preproc_path
    if "SLURM_TMPDIR" in os.environ:
        preproc_path = preproc_path.replace("$SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])
        preproc_path = preproc_path.replace("SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])
    data_file = h5py.File(preproc_path, "r")
    ood_data_list = [
        run
        for run in data_file.keys()
        if CATEGORIES["remaining"][0] in run or CATEGORIES["remaining"][1] in run
    ]
    get_activations(
        model=model,
        data_file=data_file,
        data_list=training_data_list,
        layer_list=cfg.layer_list,
        batch_size=cfg.batch_size,
        device=device,
        out_file_path=tng_h5_path,
    )
    get_activations(
        model=model,
        data_file=data_file,
        data_list=validation_data_list,
        layer_list=cfg.layer_list,
        batch_size=cfg.batch_size,
        device=device,
        out_file_path=val_h5_path,
    )
    get_activations(
        model=model,
        data_file=data_file,
        data_list=test_data_list,
        layer_list=cfg.layer_list,
        batch_size=cfg.batch_size,
        device=device,
        out_file_path=test_h5_path,
    )
    get_activations(
        model=model,
        data_file=data_file,
        data_list=ood_data_list,
        layer_list=cfg.layer_list,
        batch_size=cfg.batch_size,
        device=device,
        out_file_path=ood_h5_path,
    )

    val_bold_RSM = None
    test_bold_RSM = None
    RSA_results = {layer_name: {} for layer_name in cfg.layer_list}

    for layer_name in cfg.layer_list:
        print("--------", layer_name, "--------")
        if cfg.standard_scale_activations or cfg.n_pca is not None:
            with h5py.File(tng_h5_path, "r") as tng_file:
                tng_activations = np.concatenate(
                    [
                        tng_file[f"{layer_name}/{run}/activations"][:]
                        for run in training_data_list
                    ]
                )
        scaler = None
        if cfg.standard_scale_activations:
            print("fitting scaler...")
            scaler = StandardScaler()
            tng_activations = scaler.fit_transform(tng_activations)
        pca = None
        if cfg.n_pca is not None and cfg.n_pca < tng_activations.shape[1]:
            print("fitting PCA...")
            pca = PCA(cfg.n_pca)
            pca.fit(tng_activations)
        tng_activations, tng_bold = get_features_and_targets_prl(
            tng_h5_path, cfg.data_path, layer_name, training_data_list, scaler, pca
        )
        val_activations, val_bold = get_features_and_targets_prl(
            val_h5_path, cfg.data_path, layer_name, validation_data_list, scaler, pca
        )
        min_val_mse = float("inf")
        csv_writer = csv.DictWriter(
            open(out_dir / f"{layer_name}_scores.csv", "w"),
            fieldnames=("alpha", "tng_r2", "tng_mse", "val_r2", "val_mse"),
        )
        csv_writer.writeheader()

        for alpha in tqdm(
            [0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6], desc="fitting ridge models"
        ):
            ridge = Ridge(alpha=alpha)
            ridge.fit(tng_activations, tng_bold)
            tng_preds = ridge.predict(tng_activations)
            tng_mse = mean_squared_error(tng_bold, tng_preds)
            tng_r2 = r2_score(tng_bold, tng_preds, multioutput="raw_values")
            val_preds = ridge.predict(val_activations)
            val_mse = mean_squared_error(val_bold, val_preds)
            val_r2 = r2_score(val_bold, val_preds, multioutput="raw_values")
            val_r2_chunked = []
            for k in range(0, len(val_activations), 100):
                val_r2_chunked.append(
                    r2_score(
                        val_bold[k : k + 100],
                        val_preds[k : k + 100],
                        multioutput="raw_values",
                    )
                )

            if val_mse < min_val_mse:
                best_ridge = ridge
                best_val_r2 = val_r2
                best_val_r2_chunked = np.stack(val_r2_chunked)
                best_tng_r2 = tng_r2
                min_val_mse = val_mse
            csv_writer.writerow(
                {
                    "alpha": alpha,
                    "tng_r2": tng_r2.mean(),
                    "tng_mse": tng_mse,
                    "val_r2": val_r2.mean(),
                    "val_mse": val_mse,
                }
            )
        np.save(out_dir / f"{layer_name}_val_r2", best_val_r2)
        np.save(out_dir / f"{layer_name}_val_r2_chunked", best_val_r2_chunked)
        np.save(out_dir / f"{layer_name}_tng_r2", best_tng_r2)
        joblib.dump(best_ridge, out_dir / f"{layer_name}_ridge.joblib")
        if pca is not None:
            joblib.dump(pca, out_dir / f"{layer_name}_pca.joblib")
        if scaler is not None:
            joblib.dump(scaler, out_dir / f"{layer_name}_scaler.joblib")
        test_activations, test_bold = get_features_and_targets_prl(
            test_h5_path,
            cfg.data_path,
            layer_name,
            test_data_list,
            scaler,
            pca,
            concat=False,
        )
        test_preds = [best_ridge.predict(a) for a in test_activations]
        test_activations = np.concatenate(test_activations)
        test_act_RSM = cosine_similarity(test_activations, test_activations)
        del test_activations
        test_r2 = {}
        with h5py.File(out_dir / "test_r2.h5", "w") as outfile:
            for i in range(len(test_preds)):
                outfile.create_dataset(
                    test_data_list[i],
                    data=r2_score(
                        test_bold[i], test_preds[i], multioutput="raw_values"
                    ),
                )

        ood_activations, ood_bold = get_features_and_targets_prl(
            ood_h5_path,
            cfg.data_path,
            layer_name,
            ood_data_list,
            scaler,
            pca,
            concat=False,
        )
        ood_preds = [best_ridge.predict(a) for a in ood_activations]
        ood_activations = np.concatenate(ood_activations)
        ood_act_RSM = cosine_similarity(ood_activations, ood_activations)
        del ood_activations
        ood_r2 = {}
        with h5py.File(out_dir / "ood_r2.h5", "w") as outfile:
            for i in range(len(ood_preds)):
                outfile.create_dataset(
                    ood_data_list[i],
                    data=r2_score(ood_bold[i], ood_preds[i], multioutput="raw_values"),
                )

        # compute RSM similarity
        test_bold = np.concatenate(test_bold)
        ood_bold = np.concatenate(ood_bold)
        if val_bold_RSM is None:
            val_bold_RSM = cosine_similarity(val_bold, val_bold)
            test_bold_RSM = cosine_similarity(test_bold, test_bold)
            ood_bold_RSM = cosine_similarity(ood_bold, ood_bold)
        val_act_RSM = cosine_similarity(val_activations, val_activations)
        val_RSM_corr, val_RSM_pval = pearsonr(
            val_bold_RSM.flatten(), val_act_RSM.flatten()
        )
        test_RSM_corr, test_RSM_pval = pearsonr(
            test_bold_RSM.flatten(), test_act_RSM.flatten()
        )
        ood_RSM_corr, ood_RSM_pval = pearsonr(
            ood_bold_RSM.flatten(), ood_act_RSM.flatten()
        )
        print(
            "val RSM corr: ",
            val_RSM_corr,
            "test RSM corr: ",
            test_RSM_corr,
            "ood RSM corr: ",
            ood_RSM_corr,
        )
        RSA_results[layer_name]["val"] = {
            "pearsonr": val_RSM_corr,
            "pvalue": val_RSM_pval,
        }
        RSA_results[layer_name]["test"] = {
            "pearsonr": test_RSM_corr,
            "pvalue": test_RSM_pval,
        }
        RSA_results[layer_name]["ood"] = {
            "pearsonr": ood_RSM_corr,
            "pvalue": ood_RSM_pval,
        }
    with open(out_dir / "RSA_results.json", "w") as outfile:
        json.dump(RSA_results, outfile, indent=2)

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
