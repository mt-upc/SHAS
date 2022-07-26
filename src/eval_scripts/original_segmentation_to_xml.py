import argparse
from pathlib import Path

import yaml


def create_xml_content(
    segmentation: list[dict],
    lang_text: list[str],
    split: str,
    src_lang: str,
    tgt_lang: str,
    is_src: bool,
) -> list[str]:
    """
    Args:
        segmentation (list): content of the yaml file
        lang_text (list): content of the transcription or translation txt file
        split (str): the split name
        src_lang (str): source language id
        tgt_lang (str): target language id
        is_src (bool): whether lang_text is transcriptions

    Returns:
        xml_content (list)
    """

    xml_content = []

    xml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_content.append("<mteval>")

    if is_src:
        xml_content.append(f'<srcset setid="{split}" srclang="{src_lang}">')
    else:
        xml_content.append(
            f'<refset setid="{split}" srclang="{src_lang}" trglang="{tgt_lang}" refid="ref">'
        )

    prev_talk_id = -1
    for sgm, txt in zip(segmentation, lang_text):

        talk_id = sgm["wav"].split(".wav")[0]

        if prev_talk_id != talk_id:

            if prev_talk_id != -1:
                xml_content.append("</doc>")

            # add content (some does not matter, but is added to replicate the required format)
            xml_content.append(f'<doc docid="{talk_id}" genre="lectures">')
            xml_content.append("<keywords>does, not, matter</keywords>")
            xml_content.append("<speaker>Someone Someoneson</speaker>")
            xml_content.append(f"<talkid>{talk_id}</talkid>")
            xml_content.append("<description>Blah blah blah.</description>")
            xml_content.append("<title>Title</title>")

            seg_id = 0
            prev_talk_id = talk_id

        seg_id += 1
        xml_content.append(f'<seg id="{seg_id}">{txt}</seg>')

    xml_content.append("</doc>")
    if is_src:
        xml_content.append("</srcset>")
    else:
        xml_content.append("</refset>")
    xml_content.append("</mteval")

    return xml_content


def original_segmentation_to_xml(
    path_to_yaml: str, path_to_src_txt: str, path_to_tgt_txt: str, path_to_output: str
):
    """
    Given a segmentation yaml, and the transcriptions/translations files
    creates two xml files (one for source and one for target)
    that can be used by the mwerSegmenter tool to align the
    translations of a segmentation with the references
    """

    split = Path(path_to_yaml).stem
    src_lang = Path(path_to_src_txt).suffix
    tgt_lang = Path(path_to_tgt_txt).suffix
    path_to_output = Path(path_to_output)

    with open(path_to_yaml, "r") as yaml_file:
        segmentation = yaml.load(yaml_file, Loader=yaml.CLoader)

    with open(path_to_src_txt, "r") as src_lang_file:
        src_lang_text = src_lang_file.read().splitlines()
    if src_lang != tgt_lang:
        with open(path_to_tgt_txt, "r") as tgt_lang_file:
            tgt_lang_text = tgt_lang_file.read().splitlines()

    # remove examples with empty source or target
    src_lang_text_, tgt_lang_text_ = [], []
    for src, tgt in zip(src_lang_text, tgt_lang_text):
        if src and tgt:
            src_lang_text_.append(src)
            tgt_lang_text_.append(tgt)

    src_xml_content = create_xml_content(
        segmentation, src_lang_text_, split, src_lang, tgt_lang, True
    )
    src_path = path_to_output / f"{split}{src_lang}.xml"
    with open(src_path, "w", encoding="UTF-8") as xml_file:
        for line in src_xml_content:
            xml_file.write(line + "\n")
    print(f"Saved source transcriptions at {src_path}")

    if src_lang != tgt_lang:
        tgt_xml_content = create_xml_content(
            segmentation, tgt_lang_text_, split, src_lang, tgt_lang, False
        )
        tgt_path = path_to_output / f"{split}{tgt_lang}.xml"
        with open(tgt_path, "w", encoding="UTF-8") as xml_file:
            for line in tgt_xml_content:
                xml_file.write(line + "\n")
        print(f"Saved target translations at {tgt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_yaml",
        "-y",
        required=True,
        type=str,
        help="absolute path to the yaml of the segmentation",
    )
    parser.add_argument(
        "--path_to_src_txt",
        "-s",
        required=True,
        type=str,
        help="absolute path to the text file of the transcriptions",
    )
    parser.add_argument(
        "--path_to_tgt_txt",
        "-t",
        required=True,
        type=str,
        help="absolute path to the text file of the translations",
    )
    parser.add_argument(
        "--path_to_output",
        "-o",
        required=True,
        type=str,
        help="absolute path to the directory of the output xmls",
    )
    args = parser.parse_args()

    original_segmentation_to_xml(
        args.path_to_yaml,
        args.path_to_src_txt,
        args.path_to_tgt_txt,
        args.path_to_output,
    )
