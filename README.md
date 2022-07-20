
# SHAS: Approaching optimal Segmentation for End-to-End Speech Translation

In this repo you can find the code of the Supervised Hybrid Audio Segmentation (SHAS) method for End-to-End Speech Translation, proposed in [Tsiamas et al. (2022)](https://arxiv.org/abs/2202.04774). You can use our method with pre-trained models to segment a collection of audio files or train and fine-tune our method on your own segmented data. We provide instructions to replicate our results from the paper on MuST-C en-de and mTEDx es-en, fr-en, it-en, pt-en. You can also find easy-to-use implementations of other segmentation methods, like fixed-length, VAD, and the hybrid methods of [Potapczyk and Przybysz (2020)](https://aclanthology.org/2020.iwslt-1.9/), [Gállego et al. (2021)](https://aclanthology.org/2021.iwslt-1.11/), and [Gaido et al. (2021)](https://aclanthology.org/2021.iwslt-1.11/).

Follow the instructions [here](#usage) to segment a collection of audio files, or the instructions [here](#more-extensive-usage) to replicate the results of the paper.

## Abstract

<em>
Speech translation models are unable to directly process long audios, like TED talks, which have to be split into shorter segments. Speech translation datasets provide manual segmentations of the audios, which are not available in real-world scenarios, and existing segmentation methods usually significantly reduce translation quality at inference time. To bridge the gap between the manual segmentation of training and the automatic one at inference, we propose Supervised Hybrid Audio Segmentation (SHAS), a method that can effectively learn the optimal segmentation from any manually segmented speech corpus. First, we train a classifier to identify the included frames in a segmentation, using speech representations from a pre-trained wav2vec 2.0. The optimal splitting points are then found by a probabilistic Divide-and-Conquer algorithm that progressively splits at the frame of lowest probability until all segments are below a pre-specified length. Experiments on MuST-C and mTEDx show that the translation of the segments produced by our method approaches the quality of the manual segmentation on 5 language pairs. Namely, SHAS retains 95-98% of the manual segmentation's BLEU score, compared to the 87-93% of the best existing methods. Our method is additionally generalizable to different domains and achieves high zero-shot performance in unseen languages.
</em>

## Results

<p align="center" width="100%">
    <img width="100%" src=/readme_figures/main_results.png>
</p>

## Citation

If you find SHAS or the contents of this repo useful for your research, please consider citing:

```
@misc{tsiamas2022shas,
      title={SHAS: Approaching optimal Segmentation for End-to-End Speech Translation}, 
      author={Ioannis Tsiamas and Gerard I. Gállego and José A. R. Fonollosa and Marta R. Costa-jussà},
      year={2022},
      eprint={2202.04774},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Usage

Clone this repository to `$SHAS_ROOT`:

```bash
git clone https://github.com/mt-upc/SHAS.git ${SHAS_ROOT}    
```

Create a conda environment using the `environment.yml` file and activate it:

```bash
conda env create -f ${SHAS_ROOT}/environment.yml && \
conda activate shas
```

### Segmentation with SHAS

Download one of the available pre-trained segmentation frame classifiers required for the SHAS method:

|[English](https://drive.google.com/u/0/uc?export=download&confirm=DOjP&id=1Y7frjVkB_85snZYHTn0PQQG_kC5afoYN)|[Spanish](https://drive.google.com/u/0/uc?export=download&confirm=BlwG&id=1f73JKIv9Z7YarIHNhxABm8H3H4dXygwC)|[French](https://drive.google.com/u/0/uc?export=download&confirm=ljlD&id=14g7CjT3c9ZZC4tjjxPva1tI9tyh4hChK)|[Italian](https://drive.google.com/u/0/uc?export=download&confirm=6j54&id=1NNd07bsRD_uNi3c4yE4kmlU3X2NNCP69)|[Portuguese](https://drive.google.com/u/0/uc?export=download&confirm=TwFA&id=1JJ7prHW7S-xKD8KkOKVK0RnJzN7QOJ7h)|[Multilingual](https://drive.google.com/u/0/uc?export=download&confirm=x9hB&id=1GzwhzbHBFtwDmQPKoDOdAfESvWBrv_wB)|
|---|---|---|---|---|---|

Make sure that the audio files you want to segment are in .wav format, mono, and sampled at 16kHz. You can convert them with:

```bash
path_to_wavs=...                       # path to the audio files that will be segmented
ls ${path_to_wavs}/*.* | parallel -j 4 ffmpeg -i {} -ac 1 -ar 16000 -hide_banner -loglevel error {.}.wav
```

Segment a collection of audio files with the SHAS method. This includes inference with the classifier and application of a probabilistic Divide-and-Conquer (pDAC) algorithm:

```bash
python ${SHAS_ROOT}/src/supervised_hybrid/segment.py \
  -wavs $path_to_wavs \                       # path to the audio files that will be segmented
  -ckpt $path_to_checkpoint \                 # path to the checkpoint of a trained segmentation frame classifier
  -yaml $path_to_custom_segmentation_yaml \   # where to save the custom segmentation yaml file
  -max $max_segment_length                    # the core parameter of pDAC (in seconds, empirically values between 14-18 work well)
```

### Segmentation with other methods

Length-based (fixed-length) segmentation:

```bash
python ${SHAS_ROOT}/src/segmentation_methods/length_based.py \
  -wavs $path_to_wavs \
  -yaml $path_to_custom_segmentation_yaml \
  -n $segment_length    # (in seconds)
```

Pause-based segmentation with webrtc VAD:

```bash
python ${SHAS_ROOT}/src/segmentation_methods/pause_based.py \
  -wavs $path_to_wavs \
  -yaml $path_to_custom_segmentation_yaml \
  -l $frame_length \        # 10, 20 or 30
  -a $aggressiveness_mode   # 1, 2 or 3
```

Hybrid segmentation with either wav2vec 2.0 or VAD as pause predictor, and either the DAC or Streaming algorithms:

```bash
python ${SHAS_ROOT}/src/segmentation_methods/hybrid.py \
  -wavs $path_to_wavs \
  -yaml $path_to_custom_segmentation_yaml \
  -pause $pause_predictor \         # wav2vec or vad
  -alg $algorithm \                 # dac or strm
  -max $max_segment_length \        # (in seconds)
  -min $min_segment_length          # (in seconds) only active for the strm alg
```

## More extensive usage

Follow these steps to replicate the results of the paper. Download the MuST-C and mTEDx data, prepare them for the Segmentation Frame Classifier training, train the classifier, generate a segmentation of a test set, translate the segments with Joint Speech-to-Text models from fairseq, do hypothesis-reference alignment, and compute BLEU scores.

### Setting up the environment

Set the environment variables:

```bash
export SHAS_ROOT=...                # the path to this repo
export MUSTC_ROOT=...               # the path to save MuST-C v2.0
export MTEDX_ROOT=...               # the path to save mTEDx
export SEGM_DATASETS_ROOT=...       # the path to save the outputs of data_prep/prepare_dataset_for_segmentation
export ST_MODELS_PATH=...           # the path to the pre-trained joint-s2t models from fairseq
export RESULTS_ROOT=...             # the path to the results
export FAIRSEQ_ROOT=...             # the path to our fairseq fork
export MWERSEGMENTER_ROOT=...       # the path to the mwerSegmenter tool
```

Clone this repository to `$SHAS_ROOT`:

```bash
git clone https://github.com/mt-upc/SHAS.git ${SHAS_ROOT}    
```

If you want to evaluate a custom segmentation, the translated segments have to be aligned with the reference translations of the manual segmentation. We are using the mwerSegmenter for the alignment.
Create a secondary python2 environment for using mwerSegmenter:

```bash
conda create -n snakes27 python=2.7
```

Download mwerSegmenter at `${MWERSEGMENTER_ROOT}` and follow the instructions in `${MWERSEGMENTER_ROOT}/README` to install it:

```bash
mkdir -p $MWERSEGMENTER_ROOT
wget https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar -zxvf mwerSegmenter.tar.gz -C ${MWERSEGMENTER_ROOT}
rm -r mwerSegmenter.tar.gz
```

Create a conda environment using the `environment.yml` file and activate it:

```bash
conda env create -f ${SHAS_ROOT}/environment.yml && \
conda activate shas
```

We are using fairseq for Speech Translation. Install our fork of fairseq:

```bash
git clone -b shas https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
pip install --editable ${FAIRSEQ_ROOT}
```

Note: You can also use the latest public fairseq version, but BLEU scores will have minor differences with the ones reported in the paper.

### Data

Download MuST-C v2 en-de to `$MUSTC_ROOT`:\
The dataset is available [here](https://ict.fbk.eu/must-c/). Press the bottom ”click here to download the corpus”, and select version V2.

Download the mTEDx x-en and ASR data to `$MTEDX_ROOT`:

```bash
mkdir -p ${MTEDX_ROOT}
mkdir -p ${MTEDX_ROOT}/log_dir
for lang_pair in {es-en,fr-en,pt-en,it-en,es,fr,pt,it}; do
  wget https://www.openslr.org/resources/100/mtedx_${lang_pair}.tgz -o ${MTEDX_ROOT}/log_dir/${lang_pair} -c -b -O - | tar -xz -C ${MTEDX_ROOT}
done
```

Convert to mono and downsample at 16kHz:

```bash
ls ${MTEDX_ROOT}/*/data/{train,valid,test}/wav/*.flac | parallel -j 12 ffmpeg -i {} -ac 1 -ar 16000 -hide_banner -loglevel error {.}.wav
```

### Prepare the datasets for segmentation

We create two tsv files (talks, segments) for each triplet of dataset-lang_pair-split. These will be used during training to create training examples by random segmentation and during evaluation to create fixed segmentation for inference.

```bash
# MuST-C en-de
mkdir -p ${SEGM_DATASETS_ROOT}/MUSTC/en-de
for split in {train,dev,tst-COMMON}; do
  python ${SHAS_ROOT}/src/data_prep/prepare_dataset_for_segmentation.py \
    -y ${MUSTC_ROOT}/en-de/data/${split}/txt/${split}.yaml \
    -w ${MUSTC_ROOT}/en-de/data/${split}/wav \
    -o ${SEGM_DATASETS_ROOT}/MUSTC/en-de
done
# mTEDx
for lang_pair in {es-en,fr-en,pt-en,it-en,es-es,fr-fr,pt-pt,it-it}; do
  mkdir -p ${SEGM_DATASETS_ROOT}/mTEDx/${lang_pair}
  for split in {train,valid,test}; do
    python ${SHAS_ROOT}/src/data_prep/prepare_dataset_for_segmentation.py \
      -y ${MTEDX_ROOT}/${lang_pair}/data/${split}/txt/${split}.yaml \
      -w ${MTEDX_ROOT}/${lang_pair}/data/${split}/wav \
      -o ${SEGM_DATASETS_ROOT}/mTEDx/${lang_pair}
  done
done
```

### Download pre-trained Speech Translation models

For translating the custom segmentations we are using the [Joint Speech-to-Text models from fairseq](https://github.com/pytorch/fairseq/tree/main/examples/speech_text_joint_to_text). Download the [bilingual model trained on MuST-C en-de](https://github.com/pytorch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/ende-mustc.md) and the [multlingual model trained on mTEDx](https://github.com/pytorch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/iwslt2021.md):

```bash
# joint-s2t-mustc-en-de
en_de_model_path=${ST_MODELS_PATH}/joint-s2t-mustc-en-de
mkdir -p $en_de_model_path
for file in {checkpoint_ave_10.pt,config.yaml,src_dict.txt,dict.txt,spm.model}; do
  wget https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/${file} -O $en_de_model_path/${file}
done
# joint-s2t-multilingual
mult_model_path=${ST_MODELS_PATH}/joint-s2t-multilingual
mkdir -p $mult_model_path
for file in {checkpoint17.pt,config.yaml,tgt_dict.txt,dict.txt,spm.model}; do
  wget https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/${file} -O $mult_model_path/${file}
done
```

To generate translation with the ST models, we have to modify the path of the `spm.model` in the task configs and remove some hardcoded paths from the cfg arguments of the checkpoints.

```bash
sed -i "s+/path/spm.model+${en_de_model_path}/spm.model+" ${en_de_model_path}/config.yaml
python ${SHAS_ROOT}/src/data_prep/fix_joint_s2t_cfg.py -c ${en_de_model_path}/checkpoint_ave_10.pt
sed -i "s+/path/spm.model+${mult_model_path}/spm.model+" ${mult_model_path}/config.yaml
python ${SHAS_ROOT}/src/data_prep/fix_joint_s2t_cfg.py -c ${mult_model_path}/checkpoint17.pt
```

### Train a Segmentation Frame Classifier (SFC) model

For a monolingual model (for example on English speech):

```bash
experiment_name=en_sfc_model
python ${SHAS_ROOT}/src/supervised_hybrid/train.py \
    --datasets ${SEGM_DATASETS_ROOT}/MUSTC/en-de \
    --results_path ${RESULTS_ROOT}/supervised_hybrid \
    --model_name facebook/wav2vec2-xls-r-300m \
    --experiment_name $experiment_name \
    --train_sets tst-COMMON \
    --eval_sets dev \
    --batch_size 14 \
    --learning_rate 2.5e-4 \
    --update_freq 20 \
    --max_epochs 8 \
    --classifier_n_transformer_layers 1 \
    --wav2vec_keep_layers 15
```

For a multilingual model trained on (English, Spanish, French, Italian, Portuguese) speech:

```bash
experiment_name=mult_sfc_model
python ${SHAS_ROOT}/src/supervised_hybrid/train.py \
    --datasets ${SEGM_DATASETS_ROOT}/MUSTC/en-de,${SEGM_DATASETS_ROOT}/mTEDx/es-es,${SEGM_DATASETS_ROOT}/mTEDx/fr-fr,${SEGM_DATASETS_ROOT}/mTEDx/it-it,${SEGM_DATASETS_ROOT}/mTEDx/pt-pt \
    --results_path ${RESULTS_ROOT}/supervised_hybrid \
    --model_name facebook/wav2vec2-xls-r-300m \
    --experiment_name $experiment_name \
    --train_sets train,train,train,train,train \
    --eval_sets dev,valid,valid,valid,valid \
    --batch_size 14 \
    --learning_rate 2.5e-4 \
    --update_freq 20 \
    --max_epochs 8 \
    --classifier_n_transformer_layers 2 \
    --wav2vec_keep_layers 15
```

(The above commands assume 1 active GPU, adjust accordingly the update_freq if you are using more)

### Create a segmentation the SHAS method

Segment a collection of audio files, by doing inference with a trained Segmentation Frame Classifier and applying a probabilistic Divide-and-Conquer (pDAC) algorithm:

```bash
python ${SHAS_ROOT}/src/supervised_hybrid/segment.py \
  -wavs $path_to_wavs \                       # path to the audio files that will be segmented
  -ckpt $path_to_checkpoint \                 # path to the checkpoint of a trained segmentation frame classifier
  -yaml $path_to_custom_segmentation_yaml \   # where to save the custom segmentation yaml file
  -max $max_segment_length                    # the core parameter of pDAC (in seconds, empirically values between 14-18 work well)
```

### Translate the segments and evaluate the translations

The `eval_custom_segmentation.sh` performs the following tasks:

* (1): translates the segments using an ST model;
* (2): does hypothesis-reference alignment with mwerSegmenter;
* (3): computes scores with sacreBLEU;

```bash
bash ${SHAS_ROOT}/src/eval_scripts/eval_custom_segmentation.sh \
  $path_to_wavs \                               # path to the audio files that will be segmented
  $path_to_custom_segmentation_yaml \           # path to the custom segmentation yaml from segment.py
  $path_to_original_segmentation_yaml \         # path to the original segmentation yaml
  $path_to_original_segment_transcriptions \    # path to the text file of the original segment transcriptions
  $path_to_original_segment_translations \      # path to the text file of the original segment translations
  $src_lang \                                   # the source language id (for example: en)
  $tgt_lang \                                   # the target language id (for example: de)
  $path_to_st_model_ckpt                        # path to the checkpoint of the joint-s2t model (use the joint-s2t-mustc-en-de for en source and joint-s2t-multilingual for the rest)
```
