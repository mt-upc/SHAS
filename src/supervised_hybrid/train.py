import argparse
import warnings
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler

from constants import HIDDEN_SIZE
from data import FixedDataloaderGenerator, RandomDataloaderGenerator
from eval import eval
from models import SegmentationFrameClassifer, prepare_wav2vec

warnings.filterwarnings("ignore")


def train(args):

    experiment_name = (
        args.experiment_name
        if args.experiment_name
        else str(datetime.now().replace(microsecond=0))
        .replace(":", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    # paths
    results_path = Path(args.results_path)
    experiment_path = results_path / experiment_name
    checkpoints_path = experiment_path / "ckpts"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    if args.log_wandb:
        run = wandb.init(
            project="shas",
            config=vars(args),
            name=experiment_name,
            dir=str(experiment_path),
        )

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    n_cpu = cpu_count()
    num_workers = min(4, n_cpu // 2)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")

    # adjust batch size for number of gpus
    effective_batch_size = args.batch_size * n_gpu if n_gpu else args.batch_size

    # train dataloader
    train_dataloder_generator = RandomDataloaderGenerator(
        args.datasets,
        effective_batch_size,
        split_name=args.train_sets,
        num_workers=num_workers,
        segment_length=args.segment_length,
    )
    # eval dataloaders
    eval_dataloader_generators = [
        FixedDataloaderGenerator(
            dataset,
            effective_batch_size,
            eval_split,
            num_workers=num_workers,
            segment_length=args.segment_length,
        )
        for dataset, eval_split in zip(
            args.datasets.split(","), args.eval_sets.split(",")
        )
    ]

    # wav2vec 2.0 with (optionaly) less layers
    wav2vec_model = prepare_wav2vec(
        args.model_name, args.wav2vec_keep_layers, main_device
    )

    # init classifier
    sfc_model = SegmentationFrameClassifer(
        d_model=HIDDEN_SIZE, n_transformer_layers=args.classifier_n_transformer_layers
    ).to(main_device)

    # optionally parallel models
    if len(device_list) > 1:
        wav2vec_model = torch.nn.DataParallel(
            wav2vec_model, device_ids=device_list, output_device=main_device
        )
        sfc_model = torch.nn.DataParallel(
            sfc_model, device_ids=device_list, output_device=main_device
        )

    if args.log_wandb:
        wandb.watch(sfc_model, log="all", log_freq=args.print_every_steps)

    # get first train dataloader to approximate total steps during training
    train_dataloader = train_dataloder_generator.generate()
    total_steps_approx = int(
        args.max_epochs * len(train_dataloader) / args.update_freq * 1.01
    )

    # Adam with cosine annealing
    optimizer = AdamW(sfc_model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_steps_approx)

    global_step = 0
    for epoch in range(args.max_epochs):
        print(f"Starting epoch {epoch} ...")

        if epoch:
            train_dataloader = train_dataloder_generator.generate()

        pos_weight = 1 - train_dataloader.dataset.pos_class_percentage
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),
            reduction="none",
        )

        sfc_model.train()

        steps_in_epoch = len(train_dataloader)
        all_losses, all_preds, all_targets = [], [], []

        for step, (audio, target, in_mask, out_mask, _, _, _) in enumerate(
            iter(train_dataloader), start=1
        ):
            global_step += 1

            audio = audio.to(main_device)
            target = target.to(main_device)
            in_mask = in_mask.to(main_device)
            out_mask = out_mask.to(main_device)

            # wav2vec last hidden state
            with torch.no_grad():
                wav2vec_hidden = wav2vec_model(
                    audio, attention_mask=in_mask
                ).last_hidden_state

            logits = sfc_model(wav2vec_hidden, out_mask)

            # loss
            loss_per_point = loss_fn(logits, target)
            loss_per_point[~out_mask] = 0
            loss = loss_per_point.sum(dim=1).mean()

            # accumulate loss
            (loss / args.update_freq).backward()

            # apply optimizer
            if (not step % args.update_freq) or (step == steps_in_epoch):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # store for summary
            all_losses.append(loss.detach().cpu().numpy().item())
            all_preds.extend(
                (torch.sigmoid(logits) >= 0.5)[out_mask]
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )
            all_targets.extend(
                (target >= 0.5)[out_mask].view(-1).detach().cpu().tolist()
            )

            if (not step % args.print_every_steps) or (step == steps_in_epoch):
                f1_micro = f1_score(all_targets, all_preds, average="micro")
                f1_macro = f1_score(all_targets, all_preds, average="macro")
                avg_loss = np.mean(all_losses)
                print(
                    "[{}]: Step {}/{}, loss = {:.4f}, f1_micro {:.4f} | f1_macro {:.4f}, lr: {:.6f}".format(
                        datetime.now().time().replace(microsecond=0),
                        str(step).zfill(len(str(steps_in_epoch))),
                        steps_in_epoch,
                        avg_loss,
                        f1_micro,
                        f1_macro,
                        scheduler.get_last_lr()[0],
                    )
                )

                if args.log_wandb:
                    wandb_dict = {
                        "loss": avg_loss,
                        "f1_micro": f1_micro,
                        "f1_macro": f1_macro,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    wandb.log(wandb_dict, step=global_step)

                all_losses, all_targets, all_preds = [], [], []

            if not global_step % args.save_every_steps:
                torch.save(
                    {
                        "state_dict": sfc_model.module.state_dict()
                        if n_gpu > 1
                        else sfc_model.state_dict(),
                        "args": args,
                    },
                    checkpoints_path / f"step-{global_step}.pt",
                )

        print(
            "[{}]: Epoch {}: Starting evaliation on {} ...".format(
                datetime.now().time().replace(microsecond=0),
                epoch,
                args.eval_sets,
            )
        )
        for eval_dataloader_generator in eval_dataloader_generators:
            sfc_model.eval()
            results = eval(
                eval_dataloader_generator,
                wav2vec_model,
                sfc_model,
                main_device,
            )
            if args.log_wandb:
                wandb.log(results)
            print(results)

        torch.save(
            {
                "state_dict": sfc_model.module.state_dict()
                if n_gpu > 1
                else sfc_model.state_dict(),
                "args": args,
            },
            checkpoints_path / f"step-{global_step}.pt",
        )

    if args.log_wandb:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        "-d",
        type=str,
        help="comma separated absolute paths to specific language directions of datasets",
    )
    parser.add_argument(
        "--results_path",
        "-r",
        type=str,
        help="($RESULTS_ROOT env variable) \
            the absolute path to the results directory",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="the wav2vec 2.0 model name to be used as a backbone",
    )
    parser.add_argument(
        "--experiment_name",
        "-exp",
        type=str,
        default="",
        help="if empty, the current datetime is used",
    )
    parser.add_argument(
        "--train_sets",
        "-train",
        type=str,
        default="train",
        help="comma separated names of training sets,"
        "each value corresponds to a dataset in args.datasets",
    )
    parser.add_argument(
        "--eval_sets",
        "-eval",
        type=str,
        default="dev",
        help="comma separated names of development sets,"
        "each value corresponds to a dataset in args.datasets",
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=14, help="the batch size per gpu"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2.5e-4,
        help="the starting learning rate",
    )
    parser.add_argument("--print_every_steps", "-print", type=int, default=50)
    parser.add_argument(
        "--save_every_steps",
        "-save",
        type=int,
        default=999999,
        help="The default is to save every epoch, choose a smaller value if"
        "you want to save checkpoints more frequently.",
    )
    parser.add_argument("--max_epochs", "-max", type=int, default=8)
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="whether to log training progress with weights-and-biases",
    )
    parser.add_argument(
        "--wav2vec_keep_layers",
        "-l",
        type=int,
        default=15,
        help="the hidden representation of which wav2vec 2.0 layer to use"
        "as an input to the audio-frame-classifier"
        "layers (including and) above this one are replaced with identities",
    )
    parser.add_argument(
        "--classifier_n_transformer_layers",
        "-n",
        type=int,
        default=1,
        help="the number of transformer layers to be used"
        "in the Segmentation Frame Classifier model",
    )
    parser.add_argument(
        "--update_freq",
        "-freq",
        type=int,
        default=1,
        help="update frequency of the optimizer",
    )
    parser.add_argument(
        "--segment_length",
        "-sgm",
        type=int,
        default=20,
        help="(in seconds) length of the random segments during training"
        "and of fixed segments during inference",
    )
    parser.add_argument(
        "--eval_max_segment_length",
        type=int,
        default=18,
        help="(in seconds) max-segment-length parameter to be used"
        "by the segmentation algorithm during evaluation",
    )
    args = parser.parse_args()

    assert (
        len(args.datasets.split(","))
        == len(args.train_sets.split(","))
        == len(args.eval_sets.split(","))
    ), "Number of datasets does not match"

    train(args)
