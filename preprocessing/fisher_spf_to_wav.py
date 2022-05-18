import os
from os import path

# Script for converting the Fisher Corpus sph files to wav files.
# Requires two environment variables:
#
#   - BASE_DIR_FISHER: The root location of the fisher corpus.
#   - SCRIPT_DIR: A directory to write the processing script.
#
# This script expects a Fisher corpus directory containing fe_03_p#_sph# directories.
# It will alter the Fisher corpus directory by adding a wav subdirectory.
#
# The output script contains many individual sph2pipe commands, which can be run
# with GNU Parallel.


if "BASE_DIR_FISHER" not in os.environ:
    print(
        "Error: Environment variable BASE_DIR_FISHER for base directory of fisher corpus is not defined!"
    )
    exit()
if "SCRIPT_DIR" not in os.environ:
    print(
        "Error: Environment variable SCRIPT_DIR for processing script output is not defined!"
    )
    exit()


def makedirs(x):
    try:
        os.makedirs(x)
    except:
        pass


BASE_DIR_FISHER = os.environ["BASE_DIR_FISHER"]
WAV_DIR_FISHER = path.join(BASE_DIR_FISHER, "wav")
SCRIPT_DIR = os.environ["SCRIPT_DIR"]

makedirs(SCRIPT_DIR)
makedirs(WAV_DIR_FISHER)

fisher_dirs = [x for x in os.listdir(BASE_DIR_FISHER) if "sph" in x]

paths = []

for dir in fisher_dirs:
    audio_dir = path.join(BASE_DIR_FISHER, dir, "audio")
    for ses_id in os.listdir(audio_dir):
        ses_dir = path.join(audio_dir, ses_id)

        for sph_filename in os.listdir(ses_dir):
            if ".sph" not in sph_filename:
                continue

            paths.append(
                (
                    path.join(ses_dir, sph_filename),
                    path.join(WAV_DIR_FISHER, sph_filename.replace(".sph", ".wav")),
                )
            )

with open(path.join(SCRIPT_DIR, "sph2wav.txt"), "w") as outfile:
    for sph_path, wav_path in paths:
        outfile.write(f"sph2pipe {sph_path} {wav_path}\n")
