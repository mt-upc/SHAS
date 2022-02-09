import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import torchaudio


def fixed_length_segmentation(
    path_to_wavs: str, path_to_segmentation_yaml: str, segment_length: int
) -> None:
    """
    Produces a fixed-length segmentation of a collection of audios with segment-length seconds
    and saves the segmentation at path_to_segmentation_yaml
    """

    segments = []
    for wav_path in tqdm(sorted(list(Path(path_to_wavs).glob("*.wav")))):

        start = 0
        talk_duration = torchaudio.info(wav_path).num_frames
        sample_rate = torchaudio.info(wav_path).sample_rate
        segment_length_frames = int(segment_length * sample_rate)
        segmentation = list(range(start, talk_duration, segment_length_frames))

        # avoid creating a very short segment at the end of the audio
        if talk_duration - segmentation[-1] > 2 * sample_rate:
            segmentation.append(talk_duration)
        else:
            segmentation[-1] = talk_duration

        prev_point = start
        for point in segmentation[1:]:
            segments.append(
                {
                    "duration": round((point - prev_point) / sample_rate, 6),
                    "offset": round(prev_point / sample_rate, 6),
                    "rW": 0,
                    "uW": 0,
                    "speaker_id": "NA",
                    "wav": wav_path.name,
                }
            )
            prev_point = point

    path_to_segmentation_yaml = Path(path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(segments, f, default_flow_style=True)

    print(
        f"Saved fixed-length segmentation with segmenth-length={segment_length} at {path_to_segmentation_yaml}"
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
        "--segment_length",
        "-n",
        type=int,
        default=20,
        help="the segment length parameter of the fixed segmentation (in seconds)",
    )
    args = parser.parse_args()

    fixed_length_segmentation(
        args.path_to_wavs, args.path_to_segmentation_yaml, args.segment_length
    )
