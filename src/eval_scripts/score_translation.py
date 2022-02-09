import argparse
import json
from pathlib import Path

import sacrebleu
from sacremoses import MosesDetokenizer, MosesTokenizer


def score_translation(
    path_to_generations: str, tgt_lang: str
) -> None:

    path_to_generations = Path(path_to_generations)

    # init tokenizer and detokenizer
    mt, md = MosesTokenizer(lang=tgt_lang), MosesDetokenizer(lang=tgt_lang)

    # load reference and hypothesis
    with open(path_to_generations / "__mreference", "r", encoding="utf-8") as f:
        reference = f.read().splitlines()
    with open(path_to_generations / "__segments", "r", encoding="utf-8") as f:
        hypothesis = f.read().splitlines()

    # detokenize (have to tokenize first with the python implementation of Moses)
    hypothesis = [md.detokenize(mt.tokenize(s)) for s in hypothesis]

    assert len(reference) == len(hypothesis)

    # get bleu score
    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    ter = sacrebleu.corpus_ter(hypothesis, [reference])
    print(bleu)
    print(ter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_generations",
        "-g",
        required=True,
        type=str,
        help="absolute path to the directory of generations (after alignment with mwerSegmenter)",
    )
    parser.add_argument(
        "--tgt_lang",
        "-l",
        type=str,
        required=True,
        help="target language identifier (German: de, etc)",
    )
    args = parser.parse_args()

    score_translation(
        args.path_to_generations, args.tgt_lang
    )
