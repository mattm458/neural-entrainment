import os
import re
from functools import partial
from multiprocessing import Pool
from os import path

import pandas as pd
import parselmouth
from tqdm import tqdm
import hyphenate

from util.audio import Timestamp, get_features, ipus_to_turns, nsyl
from util.fs import makedirs

transcript_line = re.compile(r"^(\d+\.\d+) (\d+\.\d+) ([AB]): (.+)$")


def get_transcript_paths(dataset_dir):
    transcript_paths = []
    for transcript_dir in (x for x in os.listdir(dataset_dir) if "tran" in x):
        data_dir = path.join(dataset_dir, transcript_dir, "data", "trans")
        for ses_id in os.listdir(data_dir):
            ses_dir = path.join(data_dir, ses_id)
            for tran_filename in os.listdir(ses_dir):
                if ".txt" not in tran_filename:
                    continue
                transcript_paths.append(path.join(ses_dir, tran_filename))
    return transcript_paths


def filename_to_ses_id(filename):
    return int(filename[:-4].split("_")[-1])


def text_preprocess(text):
    return (
        text.replace("[mn]", "")
        .replace("(( mm ))", "mm")
        .replace("[laughter]", "")
        .replace("[noise]", "")
        .replace("((", "")
        .replace("))", "")
        .strip()
    )


def transcript_to_ipus(transcript_file):
    timestamp_data = []

    with open(transcript_file) as infile:
        wav_file = infile.readline().strip()[2:].replace(".sph", ".wav")

        for line in infile:
            matches = transcript_line.match(line)
            if matches is None:
                continue

            start, end, speaker, text = (
                matches.group(1),
                matches.group(2),
                matches.group(3),
                text_preprocess(matches.group(4)),
            )

            if text == "":
                continue
            timestamp_data.append(Timestamp(float(start), float(end), speaker, text))
    return wav_file, timestamp_data


def ipus_to_turns(timestamps):
    turn_start = None
    turn_end = None
    turn_speaker = None
    turn_text = ""

    turns = []

    for start, end, speaker, text in timestamps:
        if turn_speaker is None:
            turn_speaker = speaker
            turn_start = start
            turn_end = end
            turn_text = text

        if turn_speaker == speaker:
            turn_end = end
            turn_text += " " + text
        else:
            turns.append(Timestamp(turn_start, turn_end, turn_speaker, turn_text))
            turn_start = start
            turn_end = end
            turn_speaker = speaker
            turn_text = text

    return turns


def process_session(timestamp_data, wav_dir, audio_out_dir):
    makedirs(audio_out_dir)

    rows = []

    wav_file, timestamps = timestamp_data
    ses_id = filename_to_ses_id(wav_file)
    ses_dir = path.join(audio_out_dir, str(ses_id))
    makedirs(ses_dir)

    wav = parselmouth.Sound(path.join(wav_dir, wav_file))
    wav_a = wav.extract_left_channel()
    wav_b = wav.extract_right_channel()

    i = 0
    for start_time, end_time, speaker, transcript in timestamps:
        wav_speaker = wav_a if speaker == "A" else wav_b
        wav_part = wav_speaker.extract_part(start_time, end_time)

        out_filename = path.join(ses_dir, f"{i}.wav")
        wav_part.save(out_filename, "WAV")

        row = get_features(wav_part)

        if row is None:
            row = {}
        else:
            row["rate"] = (
                sum([nsyl(w) for w in transcript.split(" ")]) / row["duration"]
            )

        row["ses_id"] = ses_id
        row["id"] = i
        row["start_time"] = start_time
        row["end_time"] = end_time
        row["speaker"] = speaker
        row["transcript"] = transcript

        rows.append(row)

        i += 1
    return rows


def run(dataset_dir, audio_dir, feature_dir, num_processes):
    wav_dir = path.join(dataset_dir, "wav")
    ipu_dir = path.join(audio_dir, "ipus2")
    turn_dir = path.join(audio_dir, "turns2")
    tail_dir = path.join(feature_dir, "tails2")

    transcript_paths = get_transcript_paths(dataset_dir)

    p = Pool(num_processes)

    ipus = [
        x
        for x in tqdm(
            p.imap_unordered(transcript_to_ipus, transcript_paths),
            total=len(transcript_paths),
            desc="Extracting IPUs from transcripts",
        )
    ]

    turns = []
    for wav_file, timestamps in tqdm(ipus, desc="Extracting turns from IPUs"):
        turns.append((wav_file, ipus_to_turns(timestamps)))

    ipu_fn = partial(process_session, wav_dir=wav_dir, audio_out_dir=ipu_dir)
    ipu_features = []
    for rows in tqdm(
        p.imap_unordered(ipu_fn, ipus),
        total=len(ipus),
        desc="Extracting IPUs and IPU features",
    ):
        ipu_features.extend(rows)

    print("Saving IPU feature data...")
    ipu_csv_filename = path.join(feature_dir, "ipus.csv")
    pd.DataFrame(ipu_features).sort_values(["ses_id", "start_time"]).to_csv(
        ipu_csv_filename, index=None
    )

    turn_fn = partial(process_session, wav_dir=wav_dir, audio_out_dir=turn_dir)
    turn_features = []
    for rows in tqdm(
        p.imap_unordered(turn_fn, turns),
        total=len(ipus),
        desc="Extracting IPUs and IPU features",
    ):
        turn_features.extend(rows)

    print("Saving turn feature data...")
    turn_csv_filename = path.join(feature_dir, "turns.csv")
    pd.DataFrame(turn_features).sort_values(["ses_id", "start_time"]).to_csv(
        turn_csv_filename, index=None
    )