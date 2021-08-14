import argparse


parser = argparse.ArgumentParser()
# ///////// Directory Info \\\\\\\\\
parser.add_argument(
    "--work-dir",
    "-w",
    type=str,
    help="Path to directory to save all files.")
parser.add_argument(
    "--data-dir",
    "-d",
    type=str,
    default = "./dataset",
    help="Path to dataset directory.")
# ///////// Training Info \\\\\\\\\
parser.add_argument(
    "--num-epochs",
    "-e",
    default=30,
    type=int,
    help="Number of epochs.")
parser.add_argument(
    "--resume-training",
    type=int,
    default=0,
    help="Resume training from epoch.")
parser.add_argument(
    "--validation-split",
    type=float,
    default=0.1,
    help="Validation split")
parser.add_argument(
    "--batch-size",
    type=int,
    default=8)
parser.add_argument(
    "--num-frames-sequence",
    type=int,
    default=9)
parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default=1)
# ///////// Testing Info \\\\\\\\\
parser.add_argument(
    "--testing-epoch",
    type=int,
    default=29,
    help="The epoch to test.")
parser.add_argument(
    "--save-outputs",
    type=bool,
    default=True,
    help="Whether or not to save the outputs.")
# ///////// Metrics Info \\\\\\\\\
parser.add_argument(
    "--threshold-spce",
    type=float,
    default=0.25)
parser.add_argument(
    "--iou-smooth-rate",
    default=1e-6)
# ///////// Image Info \\\\\\\\\
parser.add_argument(
    "--original-image-shape",
    default=(1920,1080))
parser.add_argument(
    "--processed-image-shape",
    default=(320,128))
parser.add_argument(
    "--shuffle_size",
    default=100)
parser.add_argument(
    "--buffer_size",
    default=16)

configs = parser.parse_args()
