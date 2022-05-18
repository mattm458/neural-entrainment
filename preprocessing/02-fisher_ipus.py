import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
from os import path

import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

# Script for breaking the Fisher corpus into individual IPU wavs.
# Requires one environment variable:
#
#   - BASE_DIR_FISHER: The root location of the fisher corpus.
#
# This script expects that 01-fisher_spf_to_wav.py has been run first.


def makedirs(x):
    try:
        os.makedirs(x)
    except:
        pass


def process_wav(wav_filename, timestamps):
    wav_path = path.join(WAV_DIR_FISHER, wav_filename)
    wav, sr = librosa.load(wav_path, mono=False)

    filename_base = wav_filename.replace(".wav", "")
    filepath_base = path.join(IPU_DIR_FISHER, filename_base)
    makedirs(filepath_base)

    for i, (start, end, speaker) in enumerate(timestamps):
        start_idx = librosa.time_to_samples(start, sr=sr)
        end_idx = librosa.time_to_samples(end, sr=sr)
        channel_idx = 0 if speaker == "A" else 1

        sf.write(
            path.join(filepath_base, f"{i}.wav"),
            wav[channel_idx][start_idx:end_idx],
            samplerate=sr,
        )


if "BASE_DIR_FISHER" not in os.environ:
    print(
        "Error: Environment variable BASE_DIR_FISHER for base directory of fisher corpus is not defined!"
    )
    exit()

BASE_DIR_FISHER = os.environ["BASE_DIR_FISHER"]
WAV_DIR_FISHER = path.join(BASE_DIR_FISHER, "wav")
IPU_DIR_FISHER = path.join(BASE_DIR_FISHER, "ipus")

if __name__ == "__main__":
    makedirs(IPU_DIR_FISHER)

    tran_dirs = [x for x in os.listdir(BASE_DIR_FISHER) if "tran" in x]

    tran_line = re.compile(r"^(\d+\.\d+) (\d+\.\d+) ([AB]): .+$")

    def tran_to_timestamps(tran_data):
        timestamp_data = []
        for line in tran_data:
            matches = tran_line.match(line)
            if matches is None:
                continue

            start, end, speaker = matches.group(1), matches.group(2), matches.group(3)
            timestamp_data.append((float(start), float(end), speaker))
        return timestamp_data

    all_timestamps = []

    for dir in tqdm(tran_dirs, desc="Extracting transcript data"):
        data_dir = path.join(BASE_DIR_FISHER, dir, "data", "trans")
        for group_id in os.listdir(data_dir):
            ses_dir = path.join(data_dir, group_id)
            for tran_filename in os.listdir(ses_dir):
                if ".txt" not in tran_filename:
                    continue

                wav_filename = tran_filename.replace(".txt", ".wav")
                tran_path = path.join(ses_dir, tran_filename)
                with open(tran_path) as infile:
                    timestamps = tran_to_timestamps(infile)

                all_timestamps.append((wav_filename, timestamps))

    Parallel(n_jobs=16)(
        delayed(process_wav)(wav_filename, timestamps)
        for wav_filename, timestamps in tqdm(all_timestamps)
    )
