import argparse
from pathlib import Path

import pandas as pd
import torchaudio
import yaml

SR = 16_000

torchaudio.set_audio_backend("sox_io")


def create_talks_df(path_to_wavs: Path, segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for each talk in the split
    with talk_id, absolute path to the wav, number of segments,
    total frames (or duration) and the sample rate
    """

    talks_df = pd.DataFrame(
        columns=["id", "path", "n_segments", "total_frames", "sample_rate"]
    )

    talks_df["id"] = segments_df.talk_id.unique().tolist()
    talks_df["path"] = talks_df.apply(lambda x: path_to_wavs / f"{x['id']}.wav", axis=1)
    talks_df["n_segments"] = talks_df.apply(
        lambda x: (segments_df.talk_id == x["id"]).sum(), axis=1
    )
    talks_df["total_frames"] = talks_df.apply(
        lambda x: torchaudio.info(x["path"]).num_frames, axis=1
    )
    talks_df["sample_rate"] = talks_df.apply(
        lambda x: torchaudio.info(x["path"]).sample_rate, axis=1
    )

    return talks_df


def create_segments_df(
    path_to_yaml: Path, noise_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Reads the yaml file in a DataFrame
    where each row has timestamps (in frames) of a unique segment from talk
    """

    with open(path_to_yaml, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    segments_df = pd.DataFrame(data=data)
    segments_df.duration = segments_df.duration.astype(float)
    segments_df.offset = segments_df.offset.astype(float)

    segments_df["talk_id"] = segments_df.wav.str.split(".wav", expand=True).iloc[:, 0]

    unique_talks_ids = segments_df.talk_id.unique().tolist()
    segments_df["segm_id"] = ""
    for talk_id in unique_talks_ids:
        talk_indices = segments_df.loc[segments_df.talk_id == talk_id].index
        n_segments = len(talk_indices)
        segments_df.loc[talk_indices, "segm_id"] = [
            f"{talk_id}_{i}" for i in range(n_segments)
        ]

    segments_df["offset"] = (segments_df.offset.astype(float) * SR).astype(int)
    segments_df.rename(columns={"offset": "start"}, inplace=True)
    segments_df["duration"] = (segments_df.duration.astype(float) * SR).astype(int)
    segments_df["end"] = segments_df.start + segments_df.duration

    segments_df = segments_df[["segm_id", "talk_id", "start", "end", "duration"]]

    # remove very small segments (probably noise)
    original_len = len(segments_df)
    segments_df = segments_df.loc[(segments_df.duration / SR) > noise_threshold]
    print(
        f"Removed {original_len - len(segments_df)} segments shorter than {noise_threshold} seconds."
    )

    return segments_df


def prepare_dataset_for_segmentation(
    path_to_yaml: str, path_to_wavs: str, path_to_output_dir: str
):
    """
    Given a segmentation yaml and a directory with wavs
    creates two tsv files, one with information about each wav
    and one with information about each segment
    These are used by hybrid_supervised.data.SegmentationDataset
    to create either training examples or apply inference
    """

    path_to_yaml = Path(path_to_yaml)
    path_to_wavs = Path(path_to_wavs)
    path_to_output_dir = Path(path_to_output_dir)

    split_name = path_to_yaml.stem

    print(f"Preparing {split_name} ...")
    segments_df = create_segments_df(path_to_yaml)
    talks_df = create_talks_df(path_to_wavs, segments_df)

    talks_df.to_csv(path_to_output_dir / f"{split_name}_talks.tsv", sep="\t")
    segments_df.to_csv(path_to_output_dir / f"{split_name}_segments.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_yaml",
        "-y",
        type=str,
        required=True,
        help="absolute path to the yaml file of the segmentation",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-w",
        type=str,
        required=True,
        help="absolute path to the directory of the wavs",
    )
    parser.add_argument(
        "--path_to_output_dir",
        "-o",
        type=str,
        required=True,
        help="absolute path to the output directory",
    )

    args = parser.parse_args()

    prepare_dataset_for_segmentation(
        args.path_to_yaml, args.path_to_wavs, args.path_to_output_dir
    )
