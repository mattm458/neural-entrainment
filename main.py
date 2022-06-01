import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse

from model.entrainment import FixedHistoryEntrainmentModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Two script modes: Preprocessing and training
    subparsers = parser.add_subparsers(help="Mode", dest="mode")

    # Preprocessing arguments
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess a dataset")
    preprocess_parser.add_argument(
        "--dataset",
        type=str,
        choices=["fisher", "switchboard", "iemocap", "bmic", "cgc"],
        required=True,
    )
    preprocess_parser.add_argument(
        "--dataset-dir",
        type=str,
        help="The base directory of the dataset",
        required=True,
    )
    preprocess_parser.add_argument(
        "--audio-dir",
        type=str,
        help="The directory to hold audio files",
        required=True,
    )
    preprocess_parser.add_argument(
        "--feature-dir",
        type=str,
        help="The directory to contain extracted features",
        required=True,
    )
    preprocess_parser.add_argument(
        "--processes", type=int, help="The number of concurrent processes", default=16
    )

    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train an entrainer")
    train_parser.add_argument("--tail-length", type=int, required=True)
    train_parser.add_argument(
        "--history", type=str, choices=["speaker", "partner", "both"], required=True
    )
    train_parser.add_argument(
        "--attention", action=argparse.BooleanOptionalAction, required=True
    )
    train_parser.add_argument("--corpus-dir", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--batch-size", type=int, required=True)

    args = parser.parse_args()

    if args.mode == "preprocess":
        from data import preprocessing

        preprocessing.run(
            args.dataset,
            args.dataset_dir,
            args.audio_dir,
            args.feature_dir,
            num_processes=args.processes,
        )
    else:
        print(args)

        FixedHistoryEntrainmentModel(has_attention=False)
