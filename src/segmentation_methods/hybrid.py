import argparse
from pathlib import Path

import yaml
from constants import INPUT_SAMPLE_RATE, WAV2VEC_FRAME_LEN
from tqdm import tqdm

from utils import (
    get_talk_segments,
    get_vad_preds_for_wav,
    get_wav2vec_preds_for_wav,
    split_predictions_dac,
    split_predictions_strm,
)


def hybrid_segmentation(
    path_to_wavs: str,
    path_to_segmentation_yaml: str,
    pause_predictor: str,
    algorithm: str,
    max_segment_length: int,
    min_segment_length: int,
    vad_frame_length: int,
    vad_aggressiveness_mode: int,
    wav2vec_model_name: str,
):
    """
    Produces a hybrid segmentation of a collection of talks with (pause_predictor, algorithm)
    and saves the segmentation at ${path_to_custom_dataset}/{max_segment_length}.yaml
    """
    
    frame_length = (
        vad_frame_length if pause_predictor == "vad" else WAV2VEC_FRAME_LEN
    ) / 1000
    max_segm_len_steps = int(max_segment_length / frame_length)
    min_segm_len_steps = int(min_segment_length / frame_length)
    min_pause_len_steps = int(0.2 / frame_length)

    print(f"Initializing {pause_predictor} pause predictor")
    if pause_predictor == "wav2vec":
        import torch
        from transformers import (
            Wav2Vec2CTCTokenizer,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForCTC,
            Wav2Vec2Processor,
        )

        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name).eval().to(device)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(wav2vec_model_name)
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=INPUT_SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    elif pause_predictor == "vad":
        from webrtcvad import Vad

        vad = Vad(vad_aggressiveness_mode)

    print(f"Generating {algorithm.upper()} segmentation ...")

    all_segments = []
    for wav_path in tqdm(sorted(list(Path(path_to_wavs).glob("*.wav")))):

        if pause_predictor == "wav2vec":
            predictions = get_wav2vec_preds_for_wav(
                wav_path, model, processor, device
            )
        elif pause_predictor == "vad":
            predictions = get_vad_preds_for_wav(
                str(wav_path),
                vad,
                vad_frame_length,
            )

        if algorithm == "strm":
            splitted_predictions = split_predictions_strm(
                predictions, max_segm_len_steps, min_segm_len_steps, min_pause_len_steps
            )
        elif algorithm == "dac":
            splitted_predictions = split_predictions_dac(
                predictions, max_segm_len_steps
            )

        segments = get_talk_segments(splitted_predictions, frame_length, wav_path.name)
        all_segments.extend(segments)

    path_to_segmentation_yaml = Path(path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(all_segments, f, default_flow_style=True)

    print(
        f"Saved hybrid segmentation with pause-predictor={pause_predictor} and algorithm={algorithm} at {path_to_segmentation_yaml}"
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
        "--pause_predictor",
        "-pause",
        type=str,
        choices=["wav2vec", "vad"],
        default="vad",
        help='The model to do the frame pause predictions. \
            Choose "vad" for the predictor of FBK2021, \
            and "wav2vec" for the predictor of UPC2021',
    )
    parser.add_argument(
        "--algorithm",
        "-alg",
        type=str,
        choices=["dac", "strm"],
        default="dac",
        help='The algorithm to be used for splitting into segments based on the pause predictions. \
            Choose "dac" for the divide-and-conquer algorithm of SRPOL2020, \
            and "strm" for the streaming algorithm of FBK2021',
    )
    parser.add_argument(
        "--max_segment_length",
        "-max",
        type=int,
        default=26,
        help="The max segment length allowed in the generated segmentation (in seconds)",
    )
    parser.add_argument(
        "--min_segment_length",
        "-min",
        type=int,
        default=0,
        help="(only active for algorithm=strm) The min segment length allowed in the generated"
            "segmentation (in seconds)",
    )
    parser.add_argument(
        "--vad_frame_length",
        "-n_vad",
        type=int,
        choices=[10, 20, 30],
        default=30,
        help="(only active for pause_predictor=vad) length of a predictive frame (in milliseconds)",
    )
    parser.add_argument(
        "--vad_aggressiveness_mode",
        "-mode_vad",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="(only active for pause_predictor=vad) higher values mean more aggressive splitting",
    )
    parser.add_argument(
        "--wav2vec_model_name",
        "-m",
        type=str,
        default="facebook/wav2vec2-large-960h-lv60-self",
        help="(only active for pause_predictor=wav2vec) The wav2vec 2.0 model to be used for predictions"
            "Choose according to the source language.",
    )
    args = parser.parse_args()

    hybrid_segmentation(
        args.path_to_wavs,
        args.path_to_segmentation_yaml,
        args.pause_predictor,
        args.algorithm,
        args.max_segment_length,
        args.min_segment_length,
        args.vad_frame_length,
        args.vad_aggressiveness_mode,
        args.wav2vec_model_name,
    )
