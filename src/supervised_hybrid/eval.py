from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def infer(
    wav2vec_model,
    sfc_model,
    dataloader,
    main_device,
) -> Tuple[np.array, np.array]:
    """Does inference with the Segmentation Frame Classifier for a single wav file
    
    Args:
        wav2vec_model: an instance of a wav2vec 2.0 model
        sfc_model: an instance of a segmentation frame classifier
        dataloader: a dataloader with the FixedSegmentationDataset of a wav
        main_device: the main torch.device

    Returns:
        Tuple[np.array, np.array]: the segmentation frame probabilities for the wav
            and (optionally) their ground truth values
    """

    duration_outframes = dataloader.dataset.duration_outframes

    talk_probs = np.empty(duration_outframes)
    talk_probs[:] = np.nan
    talk_targets = np.zeros(duration_outframes)

    for audio, targets, in_mask, out_mask, included, starts, ends in iter(dataloader):

        audio = audio.to(main_device)
        in_mask = in_mask.to(main_device)
        out_mask = out_mask.to(main_device)

        with torch.no_grad():
            wav2vec_hidden = wav2vec_model(
                audio, attention_mask=in_mask
            ).last_hidden_state

            # some times the output of wav2vec is 1 frame larger/smaller
            # correct for these cases
            size1 = wav2vec_hidden.shape[1]
            size2 = out_mask.shape[1]
            if size1 != size2:
                if size1 < size2:
                    out_mask = out_mask[:, :-1]
                    ends = [e - 1 for e in ends]
                else:
                    wav2vec_hidden = wav2vec_hidden[:, :-1, :]

            logits = sfc_model(wav2vec_hidden, out_mask)
            probs = torch.sigmoid(logits)
            probs[~out_mask] = 0

        probs = probs.detach().cpu().numpy()

        # fill-in the probabilities and targets for the talk
        for i in range(len(probs)):
            start, end = starts[i], ends[i]
            if included[i] and end > start:
                duration = end - start
                talk_probs[start:end] = probs[i, :duration]
                if targets is not None:
                    talk_targets[start:end] = targets[i, :duration].numpy()
            elif not included[i]:
                talk_probs[start:end] = 0

    # account for the rare incident that a frame didnt have a prediction
    # fill-in those frames with the average of the surrounding frames
    nan_idx = np.where(np.isnan(talk_probs))[0]
    for j in nan_idx:
        talk_probs[j] = np.nanmean(
            talk_probs[max(0, j - 2) : min(duration_outframes, j + 3)]
        )

    return talk_probs, talk_targets


def eval(dataloader_generator, wav2vec_model, sfc_model, main_device) -> dict[str, float]:
    """Does inference and evaluation for a dev/test set of a language pair

    Args:
        dataloader_generator: wrapper for the individual dataloaders of each wav
        wav2vec_model: an instance of a wav2vec 2.0 model
        sfc_model: an instance of a segmentation frame classifier
        main_device: the main torch.device
        
    Returns:
        dict[str, float]: the f1 scores for this language pair
    """

    all_preds, all_targets = np.array([]), np.array([])
    talk_ids = dataloader_generator.get_talk_ids()

    for talk_id in tqdm(talk_ids):

        inference_times = dataloader_generator.dataset.inference_times
        probs, targets = None, None
        
        # multiple inferences on different fixed-length segmentations
        for iteration in range(inference_times):

            # get dataloader object for this specific segmentation of the wav
            dataloader = dataloader_generator.generate(talk_id, iteration)

            # get probabilities and targets
            p, t = infer(
                wav2vec_model,
                sfc_model,
                dataloader,
                main_device,
            )
            if probs is None:
                probs = p.copy()
                targets = t.copy()
            else:
                probs += p

        # predictions for the wav
        preds = probs / inference_times > 0.5
        all_preds = np.append(all_preds, preds)
        all_targets = np.append(all_targets, targets)

    all_targets = all_targets.astype(bool)
    all_preds = all_preds.astype(bool)

    results = {
        f"eval_f1_macro_{dataloader_generator.lang_pair}": round(
            f1_score(all_targets, all_preds, average="macro"), 4
        ),
        f"eval_f1_micro_{dataloader_generator.lang_pair}": round(
            f1_score(all_targets, all_preds, average="micro"), 4
        ),
    }

    return results
