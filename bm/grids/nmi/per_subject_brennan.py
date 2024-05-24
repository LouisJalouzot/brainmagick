"""Getting models trained on varying number of subjects."""

from itertools import product  # noqa

from .._explorers import ClipExplorer


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2,
        mem_per_gpu=100,
        partition="gpu",
    )
    # See conf/model/clip_conv.yaml for the configuration used.
    launcher.bind_({"model": "clip_conv", "optim.batch_size": 256})

    seeds = [2036]
    audio_sets = [
        "brennan2019",
    ]
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({"dset.selections": [dset]}, seed=seed)
            sub.bind_({"dset.n_subjects_test": 1})

            for n_subj in range(3, 4, 3):
                sub({"dset.n_subjects": n_subj})
