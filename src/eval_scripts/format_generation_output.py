import argparse


def format_generation_output(args) -> None:
    """
    Parses the txt with the generation output from fairseq-generate
    and creates a new txt file with only the generated translations.

    generated translations start with D-$i where $i is the correct order
    of the sentence in the dataset
    """

    raw_generation, correct_order = [], []
    with open(args.path_to_generation_file, "r", encoding="utf8") as f:
        for line in f.read().splitlines():
            if line[:2] == "D-":
                correct_order.append(int(line.split(maxsplit=1)[0].split("D-")[-1]))
                splits = line.split(maxsplit=2)
                if len(splits) == 3:
                    raw_generation.append(splits[2])
                else:
                    raw_generation.append("")

    # fix to correct order
    raw_generation = [gen for _, gen in sorted(zip(correct_order, raw_generation))]

    # save clean generation txt file
    if args.path_to_output_file is None:
        path_to_output_file = "_formatted.".join(
            args.path_to_generation_file.rsplit(".", maxsplit=1)
        )
    else:
        path_to_output_file = args.path_to_output_file
    with open(path_to_output_file, "w", encoding="utf8") as f:
        for line in raw_generation:
            f.write(line + "\n")

    print(f"Saved formatted generation at {path_to_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_generation_file",
        "-p",
        required=True,
        type=str,
        help="File to the fairseq generate.",
    )
    parser.add_argument(
        "--path_to_output_file",
        "-o",
        default=None,
        type=str,
        help="File to save the formatted translations.",
    )
    args = parser.parse_args()

    format_generation_output(args)
