# This script creates a CSV file suitable for testing a model's ability to detect
# speech features from audio data.
#
# This is an intermediary version of the script that relies on a Fisher Corpus
# SQLite database that is not generally available. In the future, it will
# use OpenSMILE features extractable through other scripts in this repository.
#
# Requires two environment variables:
#
#   - FISHER_DB: The location of the Fisher corpus database
#   - OUT_DIR: The directory to save the output CSV file
#
# This script expects that 01-fisher_spf_to_wav.py has been run first.

import os
from os import path
import sqlite3
from tqdm import tqdm
import csv


if "FISHER_DB" not in os.environ:
    print(
        "Error: Environment variable FISHER_DB for Fisher SQLite database is not defined!"
    )
    exit()


if "OUT_DIR" not in os.environ:
    print(
        "Error: Environment variable OUT_DIR for saving output CSV file is not defined!"
    )
    exit()

if __name__ == "__main__":
    with sqlite3.connect(os.environ["FISHER_DB"]) as conn:
        dataCopy = conn.execute("select count(*) from chunks")
        (length,) = dataCopy.fetchone()

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT chu.chunk_index, tur.turn_index, tsk.ses_id, chu.opensmile_features
            FROM chunks chu
            JOIN turns tur ON tur.tur_id = chu.tur_id
            JOIN tasks tsk ON tsk.tsk_id = tur.tsk_id
            ORDER BY ses_id, turn_index, chunk_index
        """
        )

        with open(path.join(os.environ["OUT_DIR"], "features.csv"), "w") as outfile:
            writer = csv.writer(outfile)

            prev_ses_id = None
            chunk_index = None
            for chunk_index, turn_index, ses_id, opensmile_features in tqdm(
                cursor, total=length
            ):
                if ses_id != prev_ses_id:
                    prev_ses_id = ses_id
                    chunk_index = -1

                chunk_index += 1

                if opensmile_features is None:
                    continue

                opensmile_features = [float(x) for x in opensmile_features.split(";")]
                writer.writerow(
                    [f"fe_03_{ses_id:05d}/{chunk_index}.wav"] + opensmile_features
                )
