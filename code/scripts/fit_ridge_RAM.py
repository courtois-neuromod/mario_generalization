import os
import csv
import h5py
import json
import joblib
import hydra
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor

from src.datasets.splitting import CATEGORIES


TR = 1.49
TIME_DOWNSAMPLE = 4
SCALER_PARTIAL_FIT_CHUNK = 100000


def get_run_features(run, data_path, states, scaler, pca):
    with h5py.File(data_path, "r") as data_file:
        onsets = data_file[f"{run}/onsets"][::TIME_DOWNSAMPLE]
        bold = data_file[f"{run}/bold"][:]
    if scaler is not None:
        states = scaler.transform(states).astype(np.float32)
    if pca is not None:
        states = pca.transform(states)
    durations = np.zeros(len(onsets))
    features = []
    for i in range(states.shape[1]):
        features.append(
            compute_regressor(
                exp_condition=[onsets, durations, states[:, i]],
                hrf_model="spm",
                frame_times=np.arange(len(bold)) * TR + TR / 2,
            )
        )
    features = np.concatenate([f[0] for f in features], axis=1)
    return features, bold


def get_features_and_targets_prl(
    data_path, run_list, states, scaler=None, pca=None, concat=True
):
    """Get features and targets for a given layer name, run list, and optional scaling/PCA."""
    features_bold = Parallel(n_jobs=-1)(
        delayed(get_run_features)(
            run_name,
            data_path,
            states[i],
            scaler,
            pca,
        )
        for i, run_name in tqdm(enumerate(run_list), desc="Generating features")
    )
    features, bold = zip(*features_bold)
    if concat:
        return np.concatenate(features), np.concatenate(bold)
    else:
        return features, bold


def filter_states(data_file, run_list, dim_filter):
    """Filter the state dimensions."""
    states = []
    run_indices = []
    index = 0
    for run in tqdm(run_list, desc="load states"):
        states.append(data_file[f"{run}/states"][::TIME_DOWNSAMPLE])
        index += states[-1].shape[0]
        run_indices.append(index)
    # the dimension filtering is a slow operation so we concat,
    # filter the concat and unconcat to do the filtering just once
    states_concat = np.concatenate(states)[:, dim_filter]
    states = []
    k = 0
    for index in run_indices:
        states.append(states_concat[k:index])
        k = index
    return states


@hydra.main(config_path="../conf", config_name="ridge", version_base=None)
def main(cfg):
    """Fit the brain encoding ridge."""
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
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
    val_bold_RSM = None
    test_bold_RSM = None
    RSA_results = {}

    all_states = np.concatenate(
        [data_file[f"{run}/states"][::TIME_DOWNSAMPLE] for run in data_file.keys()]
    )
    not_constant_states = []
    for k in tqdm(range(all_states.shape[1]), desc="Identifying constant states"):
        not_constant_states.append(len(np.unique(all_states[:, k])) > 1)
    not_constant_states = np.array(not_constant_states)
    del all_states

    tng_states = filter_states(data_file, training_data_list, not_constant_states)
    val_states = filter_states(data_file, validation_data_list, not_constant_states)
    test_states = filter_states(data_file, test_data_list, not_constant_states)
    ood_states = filter_states(data_file, ood_data_list, not_constant_states)

    print("fitting scaler...")
    scaler = StandardScaler()
    for run_states in tng_states:
        scaler = scaler.partial_fit(run_states)
    print("scaler fitted")
    pca = None
    if cfg.n_pca is not None:
        if hasattr(cfg, "random_proj") and cfg.random_proj:
            print("fitting random projection...")
            pca = GaussianRandomProjection(cfg.n_pca, random_state=cfg.seed)
            pca.fit(tng_states[0])
        else:
            print("fitting PCA...")
            pca = PCA(cfg.n_pca)
            partial_states = []
            print("len(training_data_list)", len(training_data_list))
            for i, run_states in tqdm(enumerate(tng_states), desc="fitting PCA"):
                partial_states.append(run_states)
                if (not i % 50 and i > 0) or i == len(training_data_list) - 1:
                    partial_states = np.concatenate(partial_states)
                    pca = pca.partial_fit(partial_states)
                    partial_states = []
    tng_states, tng_bold = get_features_and_targets_prl(
        cfg.data_path,
        training_data_list,
        tng_states,
        scaler,
        pca,
    )
    val_states, val_bold = get_features_and_targets_prl(
        cfg.data_path,
        validation_data_list,
        val_states,
        scaler,
        pca,
    )
    min_val_mse = float("inf")
    csv_writer = csv.DictWriter(
        open(out_dir / "RAM_scores.csv", "w"),
        fieldnames=("alpha", "tng_r2", "tng_mse", "val_r2", "val_mse"),
    )
    csv_writer.writeheader()

    for alpha in tqdm(
        [0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6], desc="fitting ridge models"
    ):
        ridge = Ridge(alpha=alpha)
        ridge.fit(tng_states, tng_bold)
        tng_preds = ridge.predict(tng_states)
        tng_mse = mean_squared_error(tng_bold, tng_preds)
        tng_r2 = r2_score(tng_bold, tng_preds, multioutput="raw_values")
        val_preds = ridge.predict(val_states)
        val_mse = mean_squared_error(val_bold, val_preds)
        val_r2 = r2_score(val_bold, val_preds, multioutput="raw_values")
        val_r2_chunked = []
        for k in range(0, len(val_states), 100):
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
    np.save(out_dir / "RAM_val_r2", best_val_r2)
    np.save(out_dir / "RAM_val_r2_chunked", best_val_r2_chunked)
    np.save(out_dir / "RAM_tng_r2", best_tng_r2)
    joblib.dump(best_ridge, out_dir / "RAM_ridge.joblib")
    if pca is not None:
        joblib.dump(pca, out_dir / "RAM_pca.joblib")
    if scaler is not None:
        joblib.dump(scaler, out_dir / "RAM_scaler.joblib")
    test_states_per_nifti, test_bold_per_nifti = get_features_and_targets_prl(
        cfg.data_path,
        test_data_list,
        test_states,
        scaler,
        pca,
        concat=False,
    )
    test_states = np.concatenate(test_states_per_nifti)
    test_bold = np.concatenate(test_bold_per_nifti)
    test_preds = best_ridge.predict(test_states)
    test_act_RSM = cosine_similarity(test_states, test_states)
    test_r2 = r2_score(test_bold, test_preds, multioutput="raw_values")
    np.save(out_dir / "RAM_test_r2", test_r2)
    test_r2_chunked = []
    for k in range(0, len(test_preds), 100):
        test_r2_chunked.append(
            r2_score(
                test_bold[k : k + 100],
                test_preds[k : k + 100],
                multioutput="raw_values",
            )
        )
    test_r2_per_nifti = []
    for k in range(len(test_states_per_nifti)):
        test_r2_per_nifti.append(
            r2_score(
                test_bold_per_nifti[k],
                best_ridge.predict(test_states_per_nifti[k]),
                multioutput="raw_values",
            )
        )
    np.save(out_dir / "RAM_test_r2_per_nifti", np.stack(test_r2_per_nifti))
    np.save(out_dir / "RAM_test_r2_chunked", np.stack(test_r2_chunked))

    ood_states_per_nifti, ood_bold_per_nifti = get_features_and_targets_prl(
        cfg.data_path,
        ood_data_list,
        ood_states,
        scaler,
        pca,
        concat=False,
    )
    ood_states = np.concatenate(ood_states_per_nifti)
    ood_bold = np.concatenate(ood_bold_per_nifti)
    ood_preds = best_ridge.predict(ood_states)
    ood_act_RSM = cosine_similarity(ood_states, ood_states)
    ood_r2 = r2_score(ood_bold, ood_preds, multioutput="raw_values")
    np.save(out_dir / "RAM_ood_r2", ood_r2)
    ood_r2_chunked = []
    for k in range(0, len(ood_preds), 100):
        ood_r2_chunked.append(
            r2_score(
                ood_bold[k : k + 100],
                ood_preds[k : k + 100],
                multioutput="raw_values",
            )
        )
    ood_r2_per_nifti = []
    for k in range(len(ood_states_per_nifti)):
        ood_r2_per_nifti.append(
            r2_score(
                ood_bold_per_nifti[k],
                best_ridge.predict(ood_states_per_nifti[k]),
                multioutput="raw_values",
            )
        )
    np.save(out_dir / "RAM_ood_r2_per_nifti", np.stack(ood_r2_per_nifti))
    np.save(out_dir / "RAM_ood_r2_chunked", np.stack(ood_r2_chunked))

    # compute RSM similarity
    if val_bold_RSM is None:
        val_bold_RSM = cosine_similarity(val_bold, val_bold)
        test_bold_RSM = cosine_similarity(test_bold, test_bold)
        ood_bold_RSM = cosine_similarity(ood_bold, ood_bold)
    val_act_RSM = cosine_similarity(val_states, val_states)
    val_RSM_corr, val_RSM_pval = pearsonr(val_bold_RSM.flatten(), val_act_RSM.flatten())
    test_RSM_corr, test_RSM_pval = pearsonr(
        test_bold_RSM.flatten(), test_act_RSM.flatten()
    )
    ood_RSM_corr, ood_RSM_pval = pearsonr(ood_bold_RSM.flatten(), ood_act_RSM.flatten())
    print(
        "val RSM corr: ",
        val_RSM_corr,
        "test RSM corr: ",
        test_RSM_corr,
        "ood RSM corr: ",
        ood_RSM_corr,
    )
    RSA_results["val"] = {
        "pearsonr": val_RSM_corr,
        "pvalue": val_RSM_pval,
    }
    RSA_results["test"] = {
        "pearsonr": test_RSM_corr,
        "pvalue": test_RSM_pval,
    }
    RSA_results["ood"] = {
        "pearsonr": ood_RSM_corr,
        "pvalue": ood_RSM_pval,
    }
    with open(out_dir / "RSA_results.json", "w") as outfile:
        json.dump(RSA_results, outfile, indent=2)


if __name__ == "__main__":
    main()
