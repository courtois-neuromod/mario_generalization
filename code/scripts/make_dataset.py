import os
import glob
import csv
import argparse
import warnings
import h5py
import retro
import numpy as np
from tqdm import tqdm
from pathlib import Path
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds

from src.ppo.emulation import replay_bk2


LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}

FWHM = 5
TR = 1.49


def main(args):
    masker = NiftiLabelsMasker(
        args.atlas_path, standardize=True, detrend=True, smoothing_fwhm=FWHM
    )
    masker.fit()
    retro.data.Integrations.add_custom_path(
        str((Path(args.source_data_dir) / "stimuli").resolve())
    )
    name = "mariostars" if args.mariostars else "mario"
    game = "SuperMarioAllStars-Snes" if args.mariostars else "SuperMarioBros-Nes"

    for sub in ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06"):
        events_path_list = glob.glob(
            f"{args.source_data_dir}/{sub}/ses-*/func/*_events.tsv"
        )
        with h5py.File(f"data/{name}_data_{sub}.h5", "w") as h5_file:
            for events_path in tqdm(events_path_list, desc=sub):
                nifti_path = events_path.replace(args.source_data_dir, args.data_dir)
                nifti_path = (
                    nifti_path.split("_run-0")[0]
                    + "_run-"
                    + nifti_path.split("_run-0")[1][0]
                    + "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                )
                if not Path(nifti_path).exists():
                    warnings.warn(
                        f"{nifti_path} doesn't exist although {events_path} does."
                    )
                    continue
                confounds, _ = load_confounds(nifti_path, **LOAD_CONFOUNDS_PARAMS)
                bold = masker.transform(nifti_path, confounds=confounds)

                with open(events_path, "r") as events_file:
                    events = csv.DictReader(events_file, delimiter="\t")
                    first_repetition = True
                    for row in events:
                        if row["trial_type"] != "gym-retro_game":
                            continue
                        if row["stim_file"] in ("Missing file", "n/a", ""):
                            first_repetition = False
                            continue
                        bk2_path = str(Path(args.source_data_dir) / row["stim_file"])
                        bk2_name = os.path.splitext(os.path.split(bk2_path)[1])[0]
                        if "SuperMarioBros-Nes" in bk2_name:  # old naming scheme
                            new_bk2_name = f"{bk2_name[:14]}_task-mario_level-w{bk2_name[-7]}l{bk2_name[-5]}_rep-{bk2_name[-3:]}"
                            bk2_path = str(
                                Path(args.source_data_dir)
                                / bk2_path.split("sourcedata/")[1].replace(
                                    bk2_name, "gamelogs/" + new_bk2_name
                                )
                            )
                            bk2_name = new_bk2_name
                        level = row["level"]
                        emulator = retro.make(
                            game,
                            inttype=retro.data.Integrations.CUSTOM_ONLY,
                            state=level,
                            render_mode=None,
                        )
                        framerate = emulator.em.get_screen_rate()
                        images, actions, done, info, states = replay_bk2(
                            bk2_path,
                            emulator,
                            skip_first_step=first_repetition,
                        )
                        emulator.close()
                        info_np = np.array([list(i.values()) for i in info])
                        first_repetition = False
                        onset = float(row["onset"])
                        bold_start_id = int(onset / TR)
                        bold_end_id = int((onset + len(images) / framerate) / TR) + 5
                        run_bold = bold[bold_start_id:bold_end_id]
                        corrected_onset = onset - bold_start_id * TR
                        onset_per_frame = (
                            np.arange(len(images)) / framerate + corrected_onset
                        )
                        group = h5_file.require_group(bk2_name)
                        group.create_dataset(
                            "frames",
                            data=images,
                            chunks=(1,) + images.shape[1:],
                            compression="gzip",
                            shuffle=True,
                            compression_opts=5,
                            dtype=np.uint8,
                        )
                        group.create_dataset("info", data=info_np)
                        group.create_dataset("onsets", data=onset_per_frame)
                        group.create_dataset("actions", data=np.array(actions))
                        group.create_dataset("done", data=np.array(done))
                        group.create_dataset("bold", data=run_bold)
                        group.create_dataset("states", data=np.stack(states))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create an hdf5 file with the actions, frames and bold data."
    )
    parser.add_argument(
        "--atlas", dest="atlas_path", type=str, required=True, help="Path to the atlas."
    )
    parser.add_argument(
        "--sourcedata",
        dest="source_data_dir",
        type=str,
        required=True,
        help="Path to the source data directory.",
    )
    parser.add_argument(
        "--datadir",
        dest="data_dir",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--mariostars",
        action="store_true",
        help="Wether the game used is MarioStars instead of SuperMarioBros",
    )
    args = parser.parse_args()
    main(args)
