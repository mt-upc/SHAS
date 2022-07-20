#!/bin/bash

path_to_wavs=$1
path_to_custom_segmentation_yaml=$2
path_to_original_segmentation_yaml=$3
path_to_original_segment_transcriptions=$4
path_to_original_segment_translations=$5
src_lang=$6
tgt_lang=$7
path_to_st_model_ckpt=$8

working_dir=$(dirname $path_to_custom_segmentation_yaml)
segmentation_name=$(basename $path_to_custom_segmentation_yaml .yaml)
split_name=$(basename $path_to_original_segmentation_yaml .yaml)
st_model_dirname=$(dirname $path_to_st_model_ckpt)
st_model_basename=$(basename $st_model_dirname)

# each model uses different inputs
if [[ $st_model_basename == "joint-s2t-mustc-en-de" ]]; then
    use_audio_input=0
elif [[ $st_model_basename == "joint-s2t-multilingual" ]]; then
    use_audio_input=1
fi

# Prepare the tsv file from the custom segmentation yaml
python ${SHAS_ROOT}/src/eval_scripts/prepare_custom_dataset.py \
    -y $path_to_custom_segmentation_yaml \
    -w $path_to_wavs \
    -l $tgt_lang \
    -i $use_audio_input

# Translate using the Speech Trasnlation model
if [[ $st_model_basename == "joint-s2t-mustc-en-de" ]]; then
    fairseq-generate $working_dir \
        --task speech_text_joint_to_text \
        --max-tokens 100000 \
        --max-source-positions 12000 \
        --nbest 1 \
        --batch-size 512 \
        --path $path_to_st_model_ckpt \
        --gen-subset $segmentation_name \
        --config-yaml ${st_model_dirname}/config.yaml \
        --scoring sacrebleu \
        --beam 5 \
        --lenpen 1.0 \
        --user-dir ${FAIRSEQ_ROOT}/examples/speech_text_joint_to_text \
        --load-speech-only > ${working_dir}/translations.txt
elif [[ $st_model_basename == "joint-s2t-multilingual" ]]; then
    fairseq-generate $working_dir \
        --task speech_text_joint_to_text \
        --user-dir ${FAIRSEQ_ROOT}/examples/speech_text_joint_to_text \
        --load-speech-only \
        --gen-subset $segmentation_name \
        --path $path_to_st_model_ckpt \
        --max-source-positions 3700000 \
        --config-yaml ${st_model_dirname}/config.yaml \
        --infer-target-lang $tgt_lang \
        --max-tokens 3700000 \
        --beam 5 > ${working_dir}/translations.txt
fi

# Extract raw hypotheses from fairseq-generate output
python ${SHAS_ROOT}/src/eval_scripts/format_generation_output.py \
    -p ${working_dir}/translations.txt

python ${SHAS_ROOT}/src/eval_scripts/original_segmentation_to_xml.py \
    -y $path_to_original_segmentation_yaml \
    -s $path_to_original_segment_transcriptions \
    -t $path_to_original_segment_translations \
    -o $working_dir

# activate the secondary python2 env
eval "$(conda shell.bash hook)"
conda activate snakes27

# align the hypotheses with the references
bash ${MWERSEGMENTER_ROOT}/segmentBasedOnMWER.sh \
    ${working_dir}/${split_name}.${src_lang}.xml \
    ${working_dir}/${split_name}.${tgt_lang}.xml \
    ${working_dir}/translations_formatted.txt \
    $st_model_basename \
    $tgt_lang \
    ${working_dir}/translations_aligned.xml \
    normalize \
    1

# re-activate main environment
eval "$(conda shell.bash hook)"
conda activate shas

# Obtain the BLEU score of the aligned hypotheses and references
python ${SHAS_ROOT}/src/eval_scripts/score_translation.py $working_dir