import collections
import contextlib
import re
import wave
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from constants import WAV2VEC_FRAME_LEN
from pydub import AudioSegment
from torch.utils.data import DataLoader, Dataset


def flatten(x: Union[list, str]) -> list[str]:
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def add_delim(x: list[str], delim: str) -> list[str]:
    x_new = [delim] * (len(x) * 2 - 1)
    x_new[0::2] = x
    return x_new


def is_pause(x: str) -> bool:
    return (set(x) == set("0")) or (x == "")


def get_pauses(pred: str) -> list[str]:
    return re.findall(r"0{1,}", pred)


def split_predictions_strm(
    preds: str, max_segm_len: int, min_segm_len: int, min_pause_len: int
) -> list[str]:
    """
    Implementation of the "Streaming" segmentation algorithm of Gaido et al, 2021
    The pause predictions are done before-hand but they are loaded in a
    streaming fashion to simulate the scenario of an audio stream

    Args:
        preds (str): binary predictions for the audio
        max_segm_len (int): maximum allowed segment length
        min_segm_len (int): minimum allowed segment length
        min_pause_len (int): minimum length of a pause

    Returns:
        list[str]: splitted binary predictions
    """

    total_duration_frames = len(preds)
    start = 0
    leftover = ""
    splitted_preds = []

    while start < total_duration_frames:

        end = min(start + max_segm_len - len(leftover), total_duration_frames)
        current_pred = leftover + preds[start:end]

        first_part = current_pred[:min_segm_len]
        second_part = current_pred[min_segm_len:]

        # find continuous pause patterns
        pauses = get_pauses(second_part)

        # get max pause
        pauses.sort(key=lambda s: len(s))
        max_pause = pauses[-1] if len(pauses) else ""

        if len(max_pause) > min_pause_len:
            first_part_b, leftover = second_part.split(max_pause, maxsplit=1)
            if is_pause(first_part):
                splitted_preds.append(first_part)
                if len(first_part_b):
                    splitted_preds.append(first_part_b)
            else:
                splitted_preds.append(first_part + first_part_b)
            splitted_preds.append(max_pause)

        else:
            splitted_preds.append(current_pred)
            leftover = ""

        start = end

    return splitted_preds


def split_predictions_dac(preds: str, max_segm_len: int) -> list[str]:
    """
    Implemntation of the divide-and-conquer algorithm of Potapczyk et al, 2020

    Args:
        preds (str): binary predictions for the audio
        max_segm_len (int): maximum allowed segment length

    Returns:
        list[str]: splitted binary predictions
    """

    preds = [preds]
    cond = [True]

    while any(cond):

        for i, current_pred in enumerate(preds):

            if cond[i]:

                # find continuous pause patterns
                pauses = get_pauses(current_pred)

                # get max pause
                pauses.sort(key=lambda s: len(s))
                max_pause = pauses[-1] if len(pauses) else ""

                if not len(max_pause):
                    preds[i] = [
                        current_pred[:max_segm_len],
                        current_pred[max_segm_len:],
                    ]
                else:
                    # split segment and add the pause between
                    preds[i] = add_delim(
                        current_pred.split(max_pause, maxsplit=1), max_pause
                    )

        preds = flatten(preds)
        preds = [pred for pred in preds if len(pred)]

        # longer than allowed and not a pause
        cond = [len(pred) > max_segm_len and not is_pause(pred) for pred in preds]

    return preds


def get_talk_segments(
    splitted_predictions: list[str], frame_length: float, wav_name: str
) -> list[dict]:
    """
    Args:
        splitted_predictions (list[str]): the splitted predictions from either STRM or DAC
        frame_length (float): the length of the predictive frame
        wav_name (str): the file name of the talk

    Returns:
        list[dict]: segmentation
    """

    # duration and offset of each pred in seconds
    durations = np.array([len(pred) for pred in splitted_predictions]) * frame_length
    durations_cumsum = np.insert(np.cumsum(durations)[1:], 0, 0)
    offsets = durations_cumsum - durations
    offsets[0] = 0

    # delete preds that are silecnes
    num_pauses = 0
    for i, preds in reversed(list(enumerate(splitted_predictions))):
        if is_pause(preds):
            del splitted_predictions[i]
            offsets = np.delete(offsets, i)
            durations = np.delete(durations, i)
            num_pauses += 1

    # expand each segment by a few seconds
    offsets = offsets - 0.06
    offsets[offsets < 0] = 0
    offsets = list(offsets)
    durations = list(durations + 0.06)

    segments = []
    for offset, duration in zip(offsets, durations):
        segments.append(
            {
                "duration": round(float(duration), 6),
                "offset": round(float(offset), 6),
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )

    return segments


class TokenPredDataset(Dataset):
    def __init__(
        self, path_to_wav: str, extra_step: float, loading_step: float
    ) -> None:
        """
        Creates a fixed-length segmentation of a wav to apply inference
        with a wav2vec 2.0 pause predictor.

        Args:
            path_to_wav (str): absolute path to the wav file
            extra_step (float): the extra length to load before and after the segments
            loading_step (float): the length of the fixed segments
        """

        super(TokenPredDataset, self).__init__()

        self.extra_step = extra_step
        self.loading_step = loading_step
        self.wav2vec_frame_length = WAV2VEC_FRAME_LEN / 1000

        # load the whole wav file
        self.wav_array, self.sr = torchaudio.backend.sox_io_backend.load(path_to_wav)
        self.wav_array = self.wav_array[0].numpy()

        self.total_duration = len(self.wav_array) // self.sr
        offset = 0
        self.start, self.end = [], []

        while offset < self.total_duration:
            if offset == 0:
                self.start.append(offset)
            else:
                # load extra_step before for more accuracy
                self.start.append(offset - self.extra_step)
            # also extra step after
            self.end.append(
                offset + self.loading_step + self.extra_step + self.wav2vec_frame_length
            )

            offset += self.loading_step

    def __len__(self) -> int:
        return len(self.start)

    def __getitem__(self, i: int) -> np.array:
        s = round(self.start[i] * self.sr)
        e = round(self.end[i] * self.sr)

        if i == len(self) - 1:
            part = self.wav_array[s:]
        else:
            part = self.wav_array[s:e]

        # normalization
        if (part != 0).sum():
            part = (part - np.mean(part)) / np.std(part)

        return part


def get_wav2vec_preds_for_wav(
    path_to_wav: str,
    model,
    processor,
    device: torch.device,
    bs: int = 8,
    loading_step: float = 10,
    extra_step: float = 1,
) -> str:
    """
    Gets binary predictions for wav file with a wav2vec 2.0 model

    Args:
        path_to_wav (str): absolute path to wav file
        model: a wav2vec 2.0 model
        processor: a wav2vec 2.0 processor
        device: a torch.device object
        bs (int, optional): Batch size. Defaults to 8.
        loading_step (float, optional): length of fixed segments. Defaults to 10.
        extra_step (float, optional): size of extra step to load before and after.
            Defaults to 1.

    Returns:
        str: binary predictions
    """

    def my_collate_fn(batch: list[np.array]) -> list[np.array]:
        return [example for example in batch]

    dataset = TokenPredDataset(path_to_wav, extra_step, loading_step)
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=my_collate_fn,
        num_workers=min(cpu_count() // 2, 4),
        drop_last=False,
    )

    # for the extra frames loaded before and after each segment
    correction = int(extra_step / dataset.wav2vec_frame_length)

    all_preds = []
    i = 0
    with torch.no_grad():
        for wav_batch in iter(dataloader):

            tokenized_audio = processor(
                wav_batch, return_tensors="pt", padding="longest", sampling_rate=16000
            )
            input_values = tokenized_audio.input_values.to(device)
            attention_mask = tokenized_audio.attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            for j, preds in enumerate(predicted_ids.tolist()):
                true_length = (
                    attention_mask[j].cpu().numpy().sum()
                    / dataset.sr
                    / dataset.wav2vec_frame_length
                )

                # apply corrections
                if i == 0:
                    preds = preds[:-correction]
                    true_length -= correction
                elif i == len(dataset) - 1:
                    preds = preds[correction:]
                    true_length -= correction
                else:
                    preds = preds[correction:-correction]
                    true_length -= 2 * correction

                # remove padding
                all_preds.extend(preds[: int(true_length)])

                i += 1

    tokens_preds = processor.tokenizer.convert_ids_to_tokens(all_preds)
    predictions = "".join(["0" if char == "<pad>" else "1" for char in tokens_preds])

    return predictions


def get_vad_preds_for_wav(talk_path: str, vad, frame_length: int):
    """
    Gets binary predictions for a wav with a VAD pause predictor

    Args:
        talk_path (str): absolute path to a wav file
        vad: a webrtc.vad object
        frame_length (int): the length of the vad precition

    Returns:
        [type]: binary predictions
    """

    audio, sample_rate = read_wave(talk_path)
    frames = frame_generator(frame_length, audio, sample_rate)
    frames = list(frames)

    predictions = "".join(
        [str(int(vad.is_speech(frame.bytes, sample_rate))) for frame in frames]
    )

    return predictions


def read_wave(path):
    """
    taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels in [1, 2]
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """
    taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    Represents a "frame" of audio data.
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(
    sample_rate, frame_duration_ms, padding_duration_ms, max_duration_sec, vad, frames
):
    """
    taken from https://github.com/wiseman/py-webrtcvad/blob/master/example.py

    Given a webrtcvad.Vad and a source of audio frames, yields the
    start and end timestamps of a voice audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD).Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    max_duration_sec - The maximum allowed duration of a segment in seconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])

            if (num_unvoiced > 0.9 * ring_buffer.maxlen) or (
                frame.timestamp + frame.duration - start
            ) > max_duration_sec:
                end = frame.timestamp + frame.duration
                triggered = False
                yield (start, end)
                ring_buffer.clear()