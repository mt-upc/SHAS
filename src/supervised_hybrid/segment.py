import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE, NOISE_THRESHOLD
from data import FixedSegmentationDatasetNoTarget, segm_collate_fn
from eval import infer
from models import SegmentationFrameClassifer, prepare_wav2vec


@dataclass
class Segment:
    def __init__(
        self,
        start: float,
        end: float,
        probs: np.array,
        decimal: int  = 4
    ) -> None:

        self.start = start if start < end else end
        self.end = end
        self.probs = probs
        self.decimal = decimal

    @property
    def duration(self):
        return float(round((self.end - self.start) / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset(self):
        return float(round(self.start / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset_plus_duration(self):
        return round(self.offset + self.duration, self.decimal)
    
    
def join(sgm_a: Segment, sgm_b: Segment) -> Segment:
    sgm = Segment(
        sgm_a.start,
        sgm_b.end,
        np.concatenate([sgm_a.probs, sgm_b.probs])
    )
    return sgm


def trim(sgm: Segment, threshold: float) -> Segment:
    """reduces the segment to between the first and last points that are above the threshold

    Args:
        sgm (Segment): a segment
        threshold (float): probability threshold

    Returns:
        Segment: new reduced segment
    """
    included_indices = np.where(sgm.probs > threshold)[0]
    
    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, sgm.probs[i:j])

    return sgm


def split(sgm: Segment, split_idx: int) -> tuple[Segment, Segment]:
    
    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    return sgm_a, sgm_b


def split_and_trim(
    sgm: Segment, split_idx: int, threshold: float
) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments

    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        threshold (float): probability threshold

    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    sgm_a, sgm_b = split(sgm, split_idx)

    sgm_a = trim(sgm_a, threshold)
    sgm_b = trim(sgm_b, threshold)

    return sgm_a, sgm_b


def pstrm(
    probs: np.array,
    max_segment_length: float,
    min_segment_length: float,
    threshold: float,
) -> list[Segment]:
    
    max_ = int(max_segment_length * TARGET_SAMPLE_RATE)
    min_ = int(min_segment_length * TARGET_SAMPLE_RATE)
    
    segments = []
    parent_sgm = Segment(0, len(probs), probs)
    parent_sgm = trim(parent_sgm, threshold)
    
    while parent_sgm.duration > 0:
        
        parent_split_idx = min(max_, parent_sgm.end)

        sgm, parent_sgm = split(parent_sgm, parent_split_idx)
        
        if sgm.duration <= min_segment_length:
            segments.append(sgm)
            continue
        
        split_idx = np.argmin(sgm.probs[min_:]) + min_
        if sgm.probs[split_idx] > threshold:
            segments.append(sgm)
        else:
            sgm_a, sgm_b = split(sgm, split_idx)
            sgm_a = trim(sgm_a, threshold)
            segments.append(sgm_a)
            
            parent_sgm = join(sgm_b, parent_sgm)

        parent_sgm = trim(parent_sgm, threshold)
        
    return segments


def pdac(
    probs: np.array,
    max_segment_length: float,
    min_segment_length: float,
    threshold: float,
    strict: bool = False
) -> list[Segment]:
    """applies the probabilistic Divide-and-Conquer algorithm to split an audio
    into segments satisfying the max-segment-length and min-segment-length conditions

    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
        threshold (float): probability threshold

    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(probs), probs)
    sgm = trim(sgm, threshold)

    def recusrive_split(sgm):
        if sgm.duration < max_segment_length:
            segments.append(sgm)
        else:
            j = 0
            sorted_indices = np.argsort(sgm.probs)
            while j < len(sorted_indices):
                split_idx = sorted_indices[j]
                split_prob = sgm.probs[split_idx]
                
                if not strict and (split_prob > threshold):
                    segments.append(sgm)
                    break

                sgm_a, sgm_b = split_and_trim(sgm, split_idx, threshold)
                if (
                    sgm_a.duration > min_segment_length
                    and sgm_b.duration > min_segment_length
                ):
                    recusrive_split(sgm_a)
                    recusrive_split(sgm_b)
                    break
                j += 1
            else:
                if not strict:
                    segments.append(sgm)
                else:
                    if (
                        sgm_a.duration > min_segment_length
                        and sgm_b.duration > min_segment_length
                    ):
                        recusrive_split(sgm_a)
                        recusrive_split(sgm_b)
                    else:
                        segments.append(sgm)

    recusrive_split(sgm)

    return segments


def update_yaml_content(
    yaml_content: list[dict], segments: list[Segment], wav_name: str
) -> list[dict]:
    """extends the yaml content with the segmentation of this wav file

    Args:
        yaml_content (list[dict]): segmentation in yaml format
        segments (list[Segment]): resulting segmentation from pdac
        wav_name (str): name of the wav file

    Returns:
        list[dict]: extended segmentation in yaml format
    """
    for sgm in segments:
        yaml_content.append(
            {
                "duration": sgm.duration,
                "offset": sgm.offset,
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )
    return yaml_content


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
    
    cache_dir = None
    if args.cache_probabilities_dir:
        cache_dir = Path(args.cache_probabilities_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    is_mustc = "MUSTC" in args.path_to_wavs
    file_paths = list(Path(args.path_to_wavs).glob("*.wav"))
    if is_mustc:
        int_ids = [int(file_path.stem.split("_")[1]) for file_path in file_paths]
        file_paths = [file_path for _, file_path in sorted(zip(int_ids, file_paths))]
    else:
        file_paths = sorted(file_paths)

    yaml_content = []
    for wav_path in tqdm(file_paths):

        if cache_dir is not None and (cache_dir / f"{wav_path.stem}.npy").is_file():
            print(f"Found cache probabilties for {wav_path.name}")
            sgm_frame_probs = np.load(cache_dir / f"{wav_path.stem}.npy")

        else:
            print(f"Doing inference with SFC for {wav_path.name}")
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
                    num_workers=4,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=segm_collate_fn,
                )

                # get frame segmentation frame probabilities in the output space
                probs, _ = infer(
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
            
            if cache_dir is not None:
                np.save(cache_dir / f"{wav_path.stem}.npy", sgm_frame_probs)

        if args.algorithm == "pdac":
            segments = pdac(
                sgm_frame_probs,
                args.dac_max_segment_length,
                args.dac_min_segment_length,
                args.dac_threshold,
                args.strict_lengths
            )
        else:
            segments = pstrm(
                sgm_frame_probs,
                args.dac_max_segment_length,
                args.dac_min_segment_length,
                args.dac_threshold,
            )
        
        segments = [sgm for sgm in segments if sgm.duration >= NOISE_THRESHOLD]

        yaml_content = update_yaml_content(yaml_content, segments, wav_path.name)

    path_to_segmentation_yaml = Path(args.path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)

    print(
        f"Saved SHAS segmentation with max={args.dac_max_segment_length} & "
        f"min={args.dac_min_segment_length} at {path_to_segmentation_yaml}"
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
        default=3,
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
    parser.add_argument(
        "--cache-probabilities-dir",
        "-cache",
        type=str,
        default="",
        help="the directory with the cache probabilities from the inference function"
    )
    parser.add_argument(
        "--strict-lengths",
        "-strict",
        action="store_true"
    )
    parser.add_argument(
        "--algorithm",
        "-alg",
        type=str,
        default="pdac",
        choices=["pdac", "pstrm"]
    )
    args = parser.parse_args()

    segment(args)
