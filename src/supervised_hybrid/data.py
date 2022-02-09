from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from constants import INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SegmentationDataset(Dataset):
    """Base class for FixedSegmentationDataset and RandomSegmentationDataset"""

    def __init__(
        self,
        path_to_dataset: str,
        split_name: str,
    ) -> None:
        """
        Args:
            path_to_dataset (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            split_name (str): name of the dataset split
        """
        super().__init__()

        self.path_to_dataset = Path(path_to_dataset)
        self.split_name = split_name

        self.input_sr = INPUT_SAMPLE_RATE
        self.target_sr = TARGET_SAMPLE_RATE
        self.in_trg_ratio = self.input_sr / self.target_sr
        self.trg_in_ratio = 1 / self.in_trg_ratio

        # load the talks and the actual segments
        self.talks_df = pd.read_csv(
            self.path_to_dataset / f"{self.split_name}_talks.tsv", sep="\t", index_col=0
        )
        self.segments_df = pd.read_csv(
            self.path_to_dataset / f"{self.split_name}_segments.tsv",
            sep="\t",
            index_col=0,
        )

        self.columns = ["talk_id", "start", "end", "duration", "included"]

        # to calculate percentage of positive examples
        self.n_pos, self.n_all = 0, 0

    def _secs_to_outframes(self, x):
        # from seconds to output space
        return np.round(x * self.target_sr).astype(int)

    def _outframes_to_inframes(self, x):
        # from output space to input space
        return np.round(x * self.in_trg_ratio).astype(int)

    def _inframes_to_outframes(self, x):
        # from input space to output space
        return np.round(x * self.trg_in_ratio).astype(int)

    def _secs_to_inframes(self, x):
        # from seconds to input space
        return np.round(x * self.input_sr).astype(int)

    def _get_targets_for_talk(self, sgm_df: pd.DataFrame, talk_id: str) -> pd.DataFrame:
        """
        Given a segmentation of a talk (sgm_df), find for
        each random segment the true_starts and true_ends that it includes.
        They are in string form separated by commas.
        Ff they are none, an empty string is passed.

        Args:
            sgm_df (pd.DataFrame): a random segmentation of a wav
            talk_id (str): unique id for the wav

        Returns:
            pd.DataFrame: sgm_df but with the 'included' column completed
        """

        true_sgm_df = self.segments_df.loc[self.segments_df.talk_id == talk_id]

        talk_targets = np.zeros(
            self.talks_df.loc[self.talks_df.id == talk_id, "total_frames"].values[0]
        )
        for idx, sgm in true_sgm_df.iterrows():
            talk_targets[sgm.start : sgm.end] = 1

        for idx, sgm in sgm_df.iterrows():

            sgm_targets = self._get_targets_for_segment(
                talk_targets[sgm.start : sgm.end]
            )
            sgm_df.loc[idx, "included"] = (
                ",".join([f"{s}:{e}" for s, e in sgm_targets]) if sgm_targets else "NA"
            )

        return sgm_df

    def _get_targets_for_segment(self, true_points: np.array) -> list[list[int]]:
        """
        Extracts the start and end points of segments in the output space
        from a binary vector defining the labels in the input space

        Args:
            true_points (np.array):
                binary label for each frame in the input space of a random segment

        Returns:
            list[list[int]]: list of tuples (start, end) in the output space
                where each tuple defines the start and end of a the true included points
        """

        points_of_change = list(np.where(true_points[1:] != true_points[:-1])[0] + 1)
        targets = []
        for s, e in zip([0] + points_of_change, points_of_change + [len(true_points)]):
            if true_points[s] == 1:
                s = self._inframes_to_outframes(s)
                e = self._inframes_to_outframes(e)

                # increase start of next segment if overlaps with end of the prev one
                if targets and s <= targets[-1][-1]:
                    s += 1

                targets.append([s, e])
                self.n_pos += e - s

        self.n_all += self._inframes_to_outframes(len(true_points))

        return targets

    def _construct_target(self, segment: pd.Series) -> torch.FloatTensor:
        """
        Given a random segment, constructs its one-hot target tensor in the output space
        """

        target_len = self._inframes_to_outframes(segment.duration)
        target = torch.zeros(target_len, dtype=torch.float)

        if segment.included != "NA":
            for s_e in segment.included.split(","):
                s, e = s_e.split(":")
                s = int(s)
                e = min(int(e), target_len + 1)
                target[s:e] = 1

        return target


class FixedSegmentationDataset(SegmentationDataset):
    def __init__(
        self,
        path_to_dataset: str,
        split_name: str,
        segment_length_secs: int = 20,
        inference_times: int = 1,
    ) -> None:
        """
        Segmentation dataset to be used during inference
        Creates a pool of examples from a fixed-length segmentation of a wav

        Args:
            path_to_dataset (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            split_name (str): name of the dataset split
            segment_length_secs (int, optional):
                The length of the fixed segments in seconds. Defaults to 20.
            inference_times (int, optional):
                How many times to perform inference on different fixed-length segmentations.
                Defaults to 1.
        """

        super().__init__(path_to_dataset, split_name)

        self.segment_length_inframes = self._secs_to_inframes(segment_length_secs)
        self.inference_times = inference_times

    def generate_fixed_segments(self, talk_id: str, i: int) -> None:
        """
        Generates a fixed-length segmentation of a wav
        with "i" controlling the begining of the segmentation
        so that different values of "i" produce different segmentations

        Args:
            talk_id (str): unique wav identifier
            i (int): indicates the current inference time
                and is used to produce a different fixed-length segmentation
                minimum allowed is 0 and maximum allowed is inference_times - 1
        """

        talk_info = self.talks_df.loc[self.talks_df["id"] == talk_id]

        self.talk_path = talk_info["path"].values[0]
        self.duration_outframes = self._inframes_to_outframes(
            self.talks_df.loc[self.talks_df["id"] == talk_id, "total_frames"].values[0]
        )
        self.duration_inframes = int(talk_info["total_frames"])

        self.fixed_segments_df = pd.DataFrame(columns=self.columns)

        start = round(self.segment_length_inframes / self.inference_times * i)
        if start > self.duration_inframes:
            start = 0
        segmentation = np.arange(
            start, self.duration_inframes, self.segment_length_inframes
        ).astype(int)
        if segmentation[0] != 0:
            segmentation = np.insert(segmentation, 0, 0)
        if segmentation[-1] != self.duration_inframes:
            if self.duration_inframes - segmentation[-1] < self._secs_to_inframes(2):
                segmentation[-1] = self.duration_inframes
            else:
                segmentation = np.append(segmentation, self.duration_inframes)

        self.fixed_segments_df["talk_id"] = talk_id
        self.fixed_segments_df["start"] = segmentation[:-1]
        self.fixed_segments_df["end"] = segmentation[1:]
        self.fixed_segments_df["duration"] = (
            self.fixed_segments_df.end - self.fixed_segments_df.start
        )

        # fill-in targets
        self.fixed_segments_df = self._get_targets_for_talk(
            self.fixed_segments_df, talk_id
        )

    def __len__(self) -> int:
        return len(self.fixed_segments_df)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
        """
        Loads the data for this fixed-length segment

        Args:
            index (int): segment id in the self.fixed_segments_df

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
                0: waveform of the segment (input space)
                1: target tensor of the segment (output space)
                2: starting frame of the segment (output space)
                3: ending frame of the segment (output space)
        """

        segment = self.fixed_segments_df.iloc[index]

        waveform, _ = torchaudio.backend.sox_io_backend.load(
            self.talk_path, frame_offset=segment.start, num_frames=segment.duration
        )

        start = self._inframes_to_outframes(segment.start + 1e-6)
        end = self._inframes_to_outframes(segment.end + 1e-6)

        target = self._construct_target(segment)

        return waveform[0], target, start, end


class RandomSegmentationDataset(SegmentationDataset):
    def __init__(
        self,
        path_to_dataset: str,
        split_name: str = "train",
        segment_length_secs: int = 20,
        seed: int = None,
    ) -> None:
        """
        Segmentation dataset to be used during training.
        Creates a pool of examples from a random segmentation of collection of wavs

        Args:
            path_to_dataset (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            split_name (str): name of the dataset split. Defaults to train.
            segment_length_secs (int, optional):
                The length of the fixed segments in seconds. Defaults to 20.
            seed (int, optional): The random seed to be used for the random segmentation.
                Defaults to None
        """

        super().__init__(path_to_dataset, split_name)

        if seed is not None:
            np.random.seed(seed)

        self.segment_length_outframes = self._secs_to_outframes(segment_length_secs)
        self.max_segment_outframes_overlap = self._secs_to_outframes(
            segment_length_secs / 10
        )
        self.segment_length_inframes = self._secs_to_inframes(segment_length_secs)

        # populate the dataset
        self.generate_random_segments()

        self.pos_class_percentage = self.n_pos / self.n_all

    def generate_random_segments(self) -> None:
        """
        Creates a new dataset by randomly segmenting each talk
        and finding the true targets that correspond to every random segment
        """

        print(
            f"Generating random segments for {self.path_to_dataset} and {self.split_name} split ..."
        )

        self.random_segments_df = pd.concat(
            [
                self._get_targets_for_talk(self._segment_talk(talk), talk["id"])
                for _, talk in tqdm(self.talks_df.iterrows())
            ],
            ignore_index=True,
        )

    def _segment_talk(self, talk: pd.Series) -> pd.DataFrame:
        """
        Produces a random segmentation of a given talk from the talks_df
        """

        rnd_sgm_df = pd.DataFrame(columns=self.columns)

        # sample in 0.02 ms but convert back to frames
        start_range = np.arange(
            0,
            self._inframes_to_outframes(talk["total_frames"]),
            step=self.segment_length_outframes - self.max_segment_outframes_overlap,
        )
        start_range = start_range - np.random.randint(
            0, self.max_segment_outframes_overlap, size=len(start_range)
        )
        start_range = self._outframes_to_inframes(start_range)

        rnd_sgm_df[["start", "end"]] = [
            (
                max(0, start),
                min(start + self.segment_length_inframes, talk["total_frames"]),
            )
            for start in start_range
        ]
        rnd_sgm_df["duration"] = rnd_sgm_df["end"] - rnd_sgm_df["start"]

        rnd_sgm_df["talk_id"] = talk["id"]

        return rnd_sgm_df

    def __len__(self) -> int:
        return len(self.random_segments_df)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        """
        Loads the data for this example of a random segment

        Args:
            index (int): the index of the random segment in the random_segments_df

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
                0: waveform of the segment (input space)
                1: target tensor of the segment (output space)
                2: starting frame of the segment (output space)
                3: ending frame of the segment (output space)
        """

        segment = self.random_segments_df.iloc[index]
        talk_path = self.talks_df.loc[
            self.talks_df.id == segment.talk_id, "path"
        ].values[0]

        # get input
        wavefrom, _ = torchaudio.backend.sox_io_backend.load(
            talk_path, frame_offset=segment.start, num_frames=segment.duration
        )

        target = self._construct_target(segment)

        start = self._inframes_to_outframes(segment.start + 1e-6)
        end = self._inframes_to_outframes(segment.end + 1e-6)

        return wavefrom[0], target, start, end


class MultRandomSegmentationDataset(RandomSegmentationDataset):
    def __init__(
        self,
        dataset_paths: list[str],
        splits: list[str],
        segment_length_secs: int = 20,
        seed: int = None,
    ) -> None:
        """
        Segmentation dataset to be used during multilingual traning.
        Creates a pool of examples by randomly segmenting many wav collections

        Args:
            path_to_dataset (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            split_name (str): name of the dataset split. Defaults to train.
            segment_length_secs (int, optional):
                The length of the fixed segments in seconds. Defaults to 20.
            seed (int, optional): The random seed to be used for the random segmentation.
                Defaults to None
        """

        # init data variables
        self.random_segments_df_parent = pd.DataFrame()
        self.talks_df_parent = pd.DataFrame()
        self.segments_df_parent = pd.DataFrame()
        self.n_pos_parent, self.n_all_parent = 0, 0

        # iterativelly populate the dataset
        for dataset_path, split in zip(dataset_paths, splits):
            super().__init__(dataset_path, split, segment_length_secs, seed)

            self.random_segments_df_parent = pd.concat(
                [self.random_segments_df_parent, self.random_segments_df],
                ignore_index=True,
            )
            self.talks_df_parent = pd.concat(
                [self.talks_df_parent, self.talks_df], ignore_index=True
            )
            self.segments_df_parent = pd.concat(
                [self.segments_df_parent, self.segments_df], ignore_index=True
            )

            self.n_pos_parent += self.n_pos
            self.n_all_parent += self.n_all

        self.pos_class_percentage = self.n_pos_parent / self.n_all_parent

        self.random_segments_df = self.random_segments_df_parent
        self.talks_df = self.talks_df_parent
        self.segments_df = self.segments_df_parent


class FixedSegmentationDatasetNoTarget(Dataset):
    def __init__(
        self,
        path_to_wav: str,
        segment_length: int = 20,
        inference_times: int = 1,
    ) -> None:
        """[summary]

        Args:
            path_to_wavs (str): [description]
            segment_length (int, optional): [description]. Defaults to 20.
            inference_times (int, optional): [description]. Defaults to 1.
        """

        super().__init__()

        self.input_sr = INPUT_SAMPLE_RATE
        self.target_sr = TARGET_SAMPLE_RATE
        self.in_trg_ratio = self.input_sr / self.target_sr
        self.trg_in_ratio = 1 / self.in_trg_ratio

        self.segment_length_inframes = self._secs_to_inframes(segment_length)
        self.inference_times = inference_times

        self.path_to_wav = path_to_wav
        self.duration_inframes = torchaudio.info(self.path_to_wav).num_frames
        self.duration_outframes = self._inframes_to_outframes(self.duration_inframes)
        self.sample_rate = torchaudio.info(self.path_to_wav).sample_rate

        assert (
            self.sample_rate == self.input_sr
        ), f"Audio needs to have sample rate of {self.input_sr}"

    def _inframes_to_outframes(self, x):
        # from input space to output space
        return np.round(x * self.trg_in_ratio).astype(int)

    def _secs_to_inframes(self, x):
        # from seconds to input space
        return np.round(x * self.input_sr).astype(int)

    def fixed_length_segmentation(self, i: int) -> None:
        """
        Generates a fixed-length segmentation of a wav
        with "i" controlling the begining of the segmentation
        so that different values of "i" produce different segmentations

        Args:
            talk_id (str): unique wav identifier
            i (int): indicates the current inference time
                and is used to produce a different fixed-length segmentation
                minimum allowed is 0 and maximum allowed is inference_times - 1
        """

        start = round(self.segment_length_inframes / self.inference_times * i)
        if start > self.duration_inframes:
            start = 0
        segmentation = np.arange(
            start, self.duration_inframes, self.segment_length_inframes
        ).astype(int)
        if segmentation[0] != 0:
            segmentation = np.insert(segmentation, 0, 0)
        if segmentation[-1] != self.duration_inframes:
            if self.duration_inframes - segmentation[-1] < self._secs_to_inframes(2):
                segmentation[-1] = self.duration_inframes
            else:
                segmentation = np.append(segmentation, self.duration_inframes)

        self.starts = segmentation[:-1]
        self.ends = segmentation[1:]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
        """
        Loads the data for this fixed-length segment

        Args:
            index (int): index of the segment in the fixed length segmentation

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
                0: waveform of the segment (input space)
                1: None for consistency with datasets that have targets
                1: starting frame of the segment (output space)
                2: ending frame of the segment (output space)
        """

        waveform, _ = torchaudio.backend.sox_io_backend.load(
            self.path_to_wav,
            frame_offset=self.starts[index],
            num_frames=self.ends[index] - self.starts[index],
        )

        start = self._inframes_to_outframes(self.starts[index] + 1e-6)
        end = self._inframes_to_outframes(self.ends[index] + 1e-6)

        return waveform[0], None, start, end


class RandomDataloaderGenerator:
    def __init__(
        self,
        dataset_roots: str,
        batch_size: int,
        split_name: str,
        num_workers: int = 0,
        segment_length: int = 20,
    ) -> None:
        """
        Helper object to be used in each epoch of training
        to produce a different random segmentation of the training data

        Args:
            dataset_roots (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            batch_size (int): training batch size (in number of examples)
            split_name (str): the name of the dataset split
            num_workers (int, optional): number of workers for the dataloader.
                Defaults to 0.
            segment_length (int, optional):
                Length of the segments (in seconds) to be produced during the random segmentation.
                Defaults to 20.
        """

        self.dataset_roots = dataset_roots
        self.num_workers = num_workers
        self.split_name = split_name
        self.batch_size = batch_size

        # for the multilingual training, dataset_roots is comma separated
        if "," in self.dataset_roots:
            self.is_mult = True
        else:
            self.is_mult = False

        self.segment_length = segment_length

        self.max_seed = 2 ** 32 - 1

    def generate(self) -> DataLoader:
        """
        Generates a random segmentation of the entire dataset
        and returns a dataloader object for it
        """

        if self.is_mult:
            dataset = MultRandomSegmentationDataset(
                self.dataset_roots.split(","),
                self.split_name.split(","),
                segment_length_secs=self.segment_length,
                seed=np.random.randint(0, self.max_seed),
            )
        else:
            dataset = RandomSegmentationDataset(
                self.dataset_roots,
                self.split_name,
                segment_length_secs=self.segment_length,
                seed=np.random.randint(0, self.max_seed),
            )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=segm_collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

        return dataloader


class FixedDataloaderGenerator:
    def __init__(
        self,
        dataset_root: str,
        batch_size: int,
        split_name: str,
        num_workers: int = 0,
        segment_length: int = 20,
        inference_times: int = 1,
    ) -> None:
        """
        Helper object to be used during inference in order to generate the
        fixed-length segmentations of a wav collection

        Args:
            dataset_roots (str): absolute path to the directory
                of _talks.tsv and _segments.tsv for the dataset
            batch_size (int): training batch size (in number of examples)
            split_name (str): the name of the dataset split
            num_workers (int, optional): number of workers for the dataloader.
                Defaults to 0.
            segment_length (int, optional):
                Length of the segments (in seconds) to be produced during the random segmentation.
                Defaults to 20.
            inference_times (int, optional):
                The number of different fixed-length segmentations to produce
                from each wav. Defaults to 1.
        """

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.lang_pair = Path(dataset_root).name

        self.dataset = FixedSegmentationDataset(
            dataset_root,
            split_name,
            segment_length_secs=segment_length,
            inference_times=inference_times,
        )

    def generate(self, talk_id: str, i: int) -> DataLoader:
        """
        Generates a fixed segmentation of a specific talk_id.
        The iteration (<= inference_times) controls the points of the fixed segmentation
        to introduce different overlaps. Returns a dataloder for this dataset.

        Args:
            talk_id (str): unique wav id
            i (int): iteration in (0, inference_times)

        Returns:
            DataLoader: a torch dataloader based on a FixedSegmentationDataset
        """

        self.dataset.generate_fixed_segments(talk_id, i)
        dataloder = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn=segm_collate_fn,
        )
        return dataloder

    def get_talk_ids(self) -> list:
        return self.dataset.talks_df["id"].tolist()


def segm_collate_fn(
    batch: list,
) -> Tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.BoolTensor,
    list[bool],
    list[int],
    list[int],
]:
    """
    (inference) collate function for the dataloader of the SegmentationDataset

    Args:
        batch (list): list of examples from SegmentationDataset

    Returns:
        Tuple[ torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor, list[bool], list[int], list[int], ]:
            0: 2D tensor, padded and normalized waveforms for each random segment
            1: 2D tensor, binary padded targets for each random segment (output space)
            2: 2D tensor, binary mask for wav2vec 2.0 (input space)
            3: 2D tensor, binary mask for audio-frame-classifier (output space)
            4: a '0' indicates that the whole example is empty (torch.zeros)
            5: the start frames of the segments (output space)
            6: the end frames of the segments (output space)
    """

    included = [bool(example[0].sum()) for example in batch]
    starts = [example[2] for example in batch]
    ends = [example[3] for example in batch]

    # sequence lengths
    in_seq_len = [len(example[0]) for example in batch]
    out_seq_len = [end - start for start, end in zip(starts, ends)]
    bs = len(in_seq_len)

    # pad and concat
    audio = torch.cat(
        [
            F.pad(example[0], (0, max(in_seq_len) - len(example[0]))).unsqueeze(0)
            for example in batch
        ]
    )

    # check if the batch contains also targets
    if batch[0][1] is not None:
        target = torch.cat(
            [
                F.pad(example[1], (0, max(out_seq_len) - len(example[1]))).unsqueeze(0)
                for example in batch
            ]
        )
    else:
        target = None

    # normalize input
    # only for inputs that have non-zero elements
    included_ = torch.tensor(included).bool()
    audio[included_] = (
        audio[included_] - torch.mean(audio[included_], dim=1, keepdim=True)
    ) / torch.std(audio[included_], dim=1, keepdim=True)

    # get masks
    in_mask = torch.ones(audio.shape, dtype=torch.long)
    out_mask = torch.ones([bs, max(out_seq_len)], dtype=torch.bool)
    for i, in_sl, out_sl in zip(range(bs), in_seq_len, out_seq_len):
        in_mask[i, in_sl:] = 0
        out_mask[i, out_sl:] = 0

    return (audio, target, in_mask, out_mask, included, starts, ends)
