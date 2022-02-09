import argparse

from pathlib import Path

import webrtcvad
import yaml
from tqdm import tqdm

from utils import read_wave, frame_generator, vad_collector


def vad_segmentation(
    path_to_wavs: str,
    path_to_segmentation_yaml: str,
    frame_length: int,
    aggressiveness_mode: int,
) -> None:
    """
    Produces a VAD segmentation of a collection of talks with webrtc's vad (aggressiveness_mode, frame_length)
    and saves the segmentation at path_to_segmentation_yaml
    """

    vad = webrtcvad.Vad(aggressiveness_mode)

    segments = []
    for wav_path in tqdm(sorted(list(Path(path_to_wavs).glob("*.wav")))):

        audio, sample_rate = read_wave(str(wav_path))
        frames = frame_generator(frame_length, audio, sample_rate)
        frames = list(frames)
        starts_ends = vad_collector(sample_rate, frame_length, 300, 60, vad, frames)

        for start, end in starts_ends:
            if end - start > 0.1:
                segments.append(
                    {
                        "duration": round(end - start, 6),
                        "offset": round(start, 6),
                        "rW": 0,
                        "uW": 0,
                        "speaker_id": "NA",
                        "wav": wav_path.name,
                    }
                )

    path_to_segmentation_yaml = Path(path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(segments, f, default_flow_style=True)

    print(
        f"Saved VAD segmentation with aggressiveness-mode={aggressiveness_mode} and frame-length={frame_length} at {path_to_segmentation_yaml}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_wavs",
        "-wavs",
        type=str,
        help="absolute path to the directory of the wav audios to be segmented",
    )
    parser.add_argument(
        "--path_to_segmentation_yaml",
        "-yaml",
        type=str,
        required=True,
        help="absolute path to the yaml file to save the generated segmentation",
    )
    parser.add_argument(
        "--frame_length",
        "-l",
        type=int,
        choices=[10, 20, 30],
        default=30,
        help="length of a predictive frame (in milliseconds)",
    )
    parser.add_argument(
        "--aggressiveness_mode",
        "-a",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="higher values mean more aggressive splitting",
    )
    args = parser.parse_args()

    vad_segmentation(
        args.path_to_wavs,
        args.path_to_segmentation_yaml,
        args.frame_length,
        args.aggressiveness_mode,
    )
