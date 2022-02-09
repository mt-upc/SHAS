import argparse

import torch


def fix_joint_s2t_cfg(path_to_ckpt):

    ckpt = torch.load(path_to_ckpt, map_location=torch.device("cpu"))

    if "mustc" in path_to_ckpt:
        ckpt["cfg"]["model"].load_pretrain_speech_encoder = ""
        ckpt["cfg"]["model"].load_pretrain_text_encoder_last = ""
        ckpt["cfg"]["model"].load_pretrain_decoder = ""
    else:
        ckpt["cfg"]["model"].load_pretrained_mbart_from = ""

    torch.save(ckpt, path_to_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_ckpt",
        "-c",
        type=str,
        required=True,
        help="absolute path to the checkpoint of the pre-trained joint-s2t model",
    )
    args = parser.parse_args()

    fix_joint_s2t_cfg(args.path_to_ckpt)
