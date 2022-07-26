# This data processing script for a customly segmented dataset is heavily based
# on the preparation scripts of fairseq speech-to-text
# https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/


import argparse
import shutil
from itertools import groupby
from pathlib import Path
from typing import Tuple

import pandas as pd
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.speech_to_text.data_utils import (convert_waveform, create_zip,
                                                extract_fbank_features,
                                                filter_manifest_df,
                                                get_zip_manifest,
                                                save_df_to_tsv)
from fairseq.data.audio.audio_utils import get_waveform

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "tgt_lang"]

SR = 16_000


class CustomDataset(Dataset):
    """
    Create a Dataset from a yaml segmentation file.
    Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    def __init__(self, path_to_yaml: str, path_to_wavs: str) -> None:

        # Load audio segments
        with open(path_to_yaml) as f:
            segments = yaml.load(f, Loader=yaml.CLoader)

        # (str -> float) to have a correct sorting of the segments for each talk
        for i, segm in enumerate(segments):
            segments[i]["offset"] = float(segm["offset"])

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = path_to_wavs / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        "NA",
                        "NA",
                        "NA",
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def prepare_custom_dataset(
    path_to_yaml: str,
    path_to_wavs: str,
    tgt_lang: str,
    use_audio_input: int,
):

    use_audio_input = bool(use_audio_input)

    path_to_yaml = Path(path_to_yaml)
    path_to_wavs = Path(path_to_wavs)

    path_to_custom_dataset = path_to_yaml.parent
    yaml_name = path_to_yaml.stem

    # Extract features
    audio_root = path_to_custom_dataset / ("flac" if use_audio_input else "fbank80")
    audio_root.mkdir(exist_ok=True, parents=True)
    zip_path = path_to_custom_dataset / f"{audio_root.name}.zip"

    dataset = CustomDataset(path_to_yaml, path_to_wavs)

    for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
        if use_audio_input:
            wf, _ = convert_waveform(
                waveform,
                sample_rate,
                to_mono=True,
                to_sample_rate=SR,
            )
            sf.write(
                audio_root / f"{utt_id}.flac",
                wf.numpy().T,
                SR,
            )
        else:
            _ = extract_fbank_features(
                waveform, sample_rate, audio_root / f"{utt_id}.npy"
            )

    # Pack features into ZIP
    print("ZIPing audios/features...")
    create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(
        zip_path,
        is_audio=use_audio_input,
    )

    # Generate TSV manifest
    print("Generating manifest...")
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for _, _, _, tgt_utt, speaker_id, utt_id in tqdm(dataset):
        manifest["id"].append(utt_id)
        manifest["audio"].append(audio_paths[utt_id])
        manifest["n_frames"].append(audio_lengths[utt_id])
        manifest["tgt_text"].append(tgt_utt)
        manifest["speaker"].append(speaker_id)
        manifest["tgt_lang"].append(tgt_lang)
    df = pd.DataFrame.from_dict(manifest)
    df = filter_manifest_df(df, is_train_split=False, is_audio=use_audio_input)
    save_df_to_tsv(df, path_to_custom_dataset / f"{yaml_name}.tsv")

    # Clean up
    shutil.rmtree(audio_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_yaml",
        "-y",
        type=str,
        required=True,
        help="absolute path to the yaml of the custom segmentation",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-w",
        type=str,
        required=True,
        help="absolute path to the directory with wavs",
    )
    parser.add_argument(
        "--tgt_lang",
        "-l",
        type=str,
        default="",
        help="optionally indicate the target language",
    )
    parser.add_argument(
        "--use_audio_input",
        "-i",
        type=int,
        default=0,
        help="whether the input is waveforms or fbank features",
    )
    args = parser.parse_args()

    prepare_custom_dataset(
        args.path_to_yaml,
        args.path_to_wavs,
        args.tgt_lang,
        args.use_audio_input,
    )
