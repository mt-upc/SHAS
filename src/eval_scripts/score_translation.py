from pathlib import Path
import sys

import sacrebleu

path_to_generations = Path(sys.argv[1])

# load reference and hypothesis
with open(path_to_generations / "__mreference", "r", encoding="utf-8") as f:
    reference = f.read().splitlines()
with open(path_to_generations / "__segments", "r", encoding="utf-8") as f:
    hypothesis = f.read().splitlines()

assert len(reference) == len(hypothesis)

# get bleu score
bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
ter = sacrebleu.corpus_ter(hypothesis, [reference])
print(bleu)
print(ter)