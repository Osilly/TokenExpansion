import argparse
import yaml
import torch
from timm.models import (
    create_model,
)

#####
import tlt.models.lvvit_expansion
from tlt.models.token_select import TokenSelect

#####
from timm.utils import *

from thop.profile import profile


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# ---------------------#
## Token Expansin
parser.add_argument("--expansion-step", nargs="+", default="[0, 100, 200]", type=int)
parser.add_argument("--keep-rate", nargs="+", default="[0.5, 0.75, 1.0]", type=float)
parser.add_argument("--initialization-keep-rate", default=0.25, type=float)
parser.add_argument("--expansion-multiple-stage", default=2, type=int)
parser.add_argument(
    "--distance",
    default="cosine",
    choices=["cosine", "manhattan", "euclidean"],
    type=str,
)
parser.add_argument("--sparse-eval", action="store_true")
# ---------------------#

parser.add_argument(
    "--model",
    default="lvvit",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "lvvit"',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        # bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        img_size=args.img_size,
    )
    # ---------------------#
    model.token_select = TokenSelect(
        expansion_step=args.expansion_step,
        keep_rate=args.keep_rate,
        initialization_keep_rate=args.initialization_keep_rate,
        expansion_multiple_stage=args.expansion_multiple_stage,
        distance=args.distance,
    )
    # ---------------------#
    model.token_select.sparse_inference = True
    model.train()

    input = torch.rand([1, 3, args.img_size, args.img_size])

    for i in range(len(args.expansion_step)):
        model.token_select.expansion_stage = i + 1
        Flops, Params = profile(model, inputs=(input,), verbose=False)
        print(
            "Stage: %d, Flops: %.2fM, Params: %.2fM"
            % (i + 1, Flops / (10**6), Params / (10**6))
        )


if __name__ == "__main__":
    main()
