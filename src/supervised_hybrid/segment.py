import argparse
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE, NOISE_THRESHOLD
from data import FixedSegmentationDatasetNoTarget, segm_collate_fn
from eval import infer
from models import SegmentationFrameClassifer, prepare_wav2vec


def trim(indices: list[int], probs: np.array, threshold: float) -> np.array:
    """trims a segment to the first and last frames i, j that have a
    segmentation frame probability greater than the threshold

    Args:
        indices (list[int]): the indices of the segment frames inside the wav
        probs (np.array): the probabilities for all the frames in the wav
        threshold (float): threshold above which a frame is classified as included

    Returns:
        np.array: the indices of the segment after trimming
    """

    included_indices = np.where(probs[indices] > threshold)[0]

    if not len(included_indices):
        return []

    real_start_idx = included_indices[0]
    real_end_idx = included_indices[-1] + 1

    indices = indices[real_start_idx:real_end_idx]

    return indices


def flatten(x: Union[list[list[int]], list[int]]) -> list[list[int]]:
    """recursivelly flattens the collection of segments"""
    if isinstance(x[0], list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def split_and_trim(
    current_indices: list[int], probs: np.array, split_idx: int, threshold: float
) -> Tuple[list[int], list[int]]:
    """splits the segment at the specified index and trims the resulting segments

    Args:
        indices (list[int]): the indices of the segment frames inside the wav
        probs (np.array): the probabilities for all the frames in the wav
        split_idx (int): the index at which to split the segment
        threshold (float): threshold above which a frame is classified as included

    Returns:
        Tuple[list[int], list[int]]: the trimmed indices of the two new segments
    """
    new_indices_a = trim(current_indices[:split_idx], probs, threshold)
    new_indices_b = trim(current_indices[split_idx + 1 :], probs, threshold)
    return new_indices_a, new_indices_b


def probabilistic_dac(
    probs: np.array,
    max_segment_length: float,
    min_segment_length: float,
    threshold: float,
) -> list[list[int]]:
    """Probabilistic Divide-and-Conquer algorithm.
    It progressively splits at the frame of lowest probability until all segments
    are below a max-segment-length.

    Args:
        probs (np.array): the probabilities for all the frames in the wav
        max_segment_length (float): Maximum allowed segment length for the potential segments
            In seconds.
        min_segment_length (float): Minimum allowed segment length for the potential segments
            In seconds.
        threshold (float): threshold above which a frame is classified as included
            Used for trimming.

    Returns:
        list[list[int]]: The resulting segments. Each segment is a list of indices (frames)
            in the wav file.
    """

    size = len(probs)
    indices = list(range(size))

    # trim silent beginning and end
    indices = trim(indices, probs, threshold)

    # start with a single segment
    indices = [indices]

    cond = [True]
    while any(cond):

        for i, current_indices in enumerate(indices):

            if cond[i]:

                # find point of highest split probability
                # and get the indices of the new segments after trimming
                split_idx = probs[current_indices].argmin()
                new_indices_a, new_indices_b = split_and_trim(
                    current_indices, probs, split_idx, threshold
                )

                # check if the resulting segments are above the min_segment_length
                j, sorted_probs_indices = 1, None
                while (
                    len(new_indices_a) / TARGET_SAMPLE_RATE < min_segment_length
                    or len(new_indices_b) / TARGET_SAMPLE_RATE < min_segment_length
                ) and j + 1 < len(current_indices):
                    j += 1

                    # sort if this is the first re-try
                    if sorted_probs_indices is None:
                        sorted_probs_indices = np.argsort(probs[current_indices])

                    # try splitting in this index
                    split_idx = sorted_probs_indices[j]
                    new_indices_a, new_indices_b = split_and_trim(
                        current_indices, probs, split_idx, threshold
                    )

                # if none of the indices satisfied the min conditions
                # split at the initial point of the lowest probability
                if j + 1 == len(current_indices):
                    split_idx = probs[current_indices].argmin()
                    new_indices_a, new_indices_b = split_and_trim(
                        current_indices, probs, split_idx, threshold
                    )

                # replace the previous segment with the two new ones
                indices[i] = []
                if new_indices_a:
                    indices[i].append(new_indices_a)
                if new_indices_b:
                    indices[i].append(new_indices_b)

        # to list of lists
        indices = flatten(indices)

        # check if max_segment_length conditions is satisfied for every segment
        cond = [len(ind) / TARGET_SAMPLE_RATE > max_segment_length for ind in indices]

    return indices


def produce_segmentation(indices: list[list[int]], wav_name: str) -> list[dict]:
    """produces the segmentation yaml content from the indices of the probabilistic_dac

    Args:
        indices (list[list[int]]): output of the probabilistic_dac function
        wav_name (str): the name of the wav file (with the .wav suffix)

    Returns:
        list[dict]: the content of the segmentation yaml
    """
    talk_segments = []
    for ind in indices:
        size = len(ind) / TARGET_SAMPLE_RATE
        if size < NOISE_THRESHOLD:
            continue
        start = ind[0] / TARGET_SAMPLE_RATE
        talk_segments.append(
            {
                "duration": round(size, 6),
                "offset": round(start, 6),
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )
    return talk_segments


def segment(args):

    device = (
        torch.device(f"cuda:0")
        if torch.cuda.device_count() > 0
        else torch.device("cpu")
    )

    checkpoint = torch.load(args.path_to_checkpoint, map_location=device)

    # init wav2vec 2.0
    wav2vec_model = prepare_wav2vec(
        checkpoint["args"].model_name,
        checkpoint["args"].wav2vec_keep_layers,
        device,
    )
    # init segmentation frame classifier
    sfc_model = SegmentationFrameClassifer(
        d_model=HIDDEN_SIZE,
        n_transformer_layers=checkpoint["args"].classifier_n_transformer_layers,
    ).to(device)
    sfc_model.load_state_dict(checkpoint["state_dict"])
    sfc_model.eval()

    all_segments = []
    for wav_path in tqdm(sorted(list(Path(args.path_to_wavs).glob("*.wav")))):

        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path, args.inference_segment_length, args.inference_times
        )
        sgm_frame_probs = None

        for inference_iteration in range(args.inference_times):

            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
            dataloader = DataLoader(
                dataset,
                batch_size=args.inference_batch_size,
                num_workers=max(cpu_count() // 2, 4),
                shuffle=False,
                drop_last=False,
                collate_fn=segm_collate_fn,
            )

            # get frame segmentation frame probabilities in the output space
            probs, _ = infer(
                wav_path.name,
                wav2vec_model,
                sfc_model,
                dataloader,
                device,
            )
            if sgm_frame_probs is None:
                sgm_frame_probs = probs.copy()
            else:
                sgm_frame_probs += probs

        sgm_frame_probs /= args.inference_times
        
        # apply the probabilistic dac to the segmentation frame probabilities
        indices = probabilistic_dac(
            sgm_frame_probs,
            args.dac_max_segment_length,
            args.dac_min_segment_length,
            args.dac_threshold,
        )
        segments = produce_segmentation(indices, wav_path.name)
        all_segments.extend(segments)

    path_to_segmentation_yaml = Path(args.path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(all_segments, f, default_flow_style=True)

    print(
        f"Saved hybrid-supervised segmentation with max-segmenth-length={args.dac_max_segment_length} at {path_to_segmentation_yaml}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_segmentation_yaml",
        "-yaml",
        type=str,
        required=True,
        help="absolute path to the yaml file to save the generated segmentation",
    )
    parser.add_argument(
        "--path_to_checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="absolute path to the audio-frame-classifier checkpoint",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-wavs",
        type=str,
        help="absolute path to the directory of the wav audios to be segmented",
    )
    parser.add_argument(
        "--inference_batch_size",
        "-bs",
        type=int,
        default=12,
        help="batch size (in examples) of inference with the audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_segment_length",
        "-len",
        type=int,
        default=20,
        help="segment length (in seconds) of fixed-length segmentation during inference"
            "with audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_times",
        "-n",
        type=int,
        default=1,
        help="how many times to apply inference on different fixed-length segmentations"
            "of each wav",
    )
    parser.add_argument(
        "--dac_max_segment_length",
        "-max",
        type=float,
        default=18,
        help="the segmentation algorithm splits until all segments are below this value"
            "(in seconds)",
    )
    parser.add_argument(
        "--dac_min_segment_length",
        "-min",
        type=float,
        default=0.2,
        help="a split by the algorithm is carried out only if the resulting two segments"
            "are above this value (in seconds)",
    )
    parser.add_argument(
        "--dac_threshold",
        "-thr",
        type=float,
        default=0.5,
        help="after each split by the algorithm, the resulting segments are trimmed to"
            "the first and last points that corresponds to a probability above this value",
    )
    args = parser.parse_args()

    segment(args)
