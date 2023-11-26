import argparse

from thop.profile import profile
from timm.models import create_model
import torch

import models.deit_expansion
from models.token_select import TokenSelect


def get_args_parser():
    parser = argparse.ArgumentParser(
        "EfficientTrain training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--expansion-step", nargs="+", default="[0, 100, 200]", type=int
    )
    parser.add_argument(
        "--keep-rate", nargs="+", default="[0.5, 0.75, 1.0]", type=float
    )
    parser.add_argument("--initialization-keep-rate", default=0.4, type=float)
    parser.add_argument("--expansion-multiple-stage", default=5, type=int)
    parser.add_argument("--sparse-eval", action="store_true")

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--bce-loss", action="store_true")
    parser.add_argument("--unscale-lr", action="store_true")

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument("--patch-size", default=16, type=int, help="images patch size")

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--nb-classes",
        type=int,
        default=1000,
    )

    return parser


def main(args):
    for i in range(5):
        if i < 2:
            input_size = 160
        elif i < 4:
            input_size = 192
        else:
            input_size = 224

        input = torch.rand([1, 3, input_size, input_size])

        model = create_model(
            args.model,
            img_size=input_size,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
        model.token_select = TokenSelect(
            expansion_step=args.expansion_step,
            keep_rate=args.keep_rate,
            initialization_keep_rate=args.initialization_keep_rate,
            expansion_multiple_stage=args.expansion_multiple_stage,
        )
        model.token_select.sparse_inference = True
        model.train()
        if i == 0:
            model.token_select.expansion_stage = 1
        elif i < 3:
            model.token_select.expansion_stage = 2
        else:
            model.token_select.expansion_stage = 3
        epochs = [100, 80, 20, 40, 60]
        Flops, Params = profile(model, inputs=(input,), verbose=False)
        print(
            "Stage: %d, Epoch number %d, Input size %d, toe_stage %d, Flops: %.2fM, Params: %.2fM"
            % (
                i + 1,
                epochs[i],
                input_size,
                model.token_select.expansion_stage,
                Flops / (10**6),
                Params / (10**6),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "EfficientTrain get Flops and Params script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
