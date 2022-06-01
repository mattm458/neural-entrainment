from os import path

from data.preprocessing import fisher
from util.fs import makedirs


def run(dataset, dataset_dir, audio_dir, feature_dir, num_processes=1):
    ipu_dir = path.join(audio_dir, "ipus")
    turn_dir = path.join(audio_dir, "turns")
    tail_dir = path.join(feature_dir, "tails")

    makedirs(ipu_dir)
    makedirs(turn_dir)
    makedirs(tail_dir)

    if dataset == "fisher":
        return fisher.run(dataset_dir, audio_dir, feature_dir, num_processes)
    else:
        raise NotImplementedError(f"Preprocessor for {dataset} not implemented")
