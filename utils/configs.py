import argparse

parser = argparse.ArgumentParser()
# Directory Info
parser.add_argument(
    "--work-dir", "-w", type=str, help="Path to directory to save all files.")
parser.add_argument(
    "--data-dir", "-d", type=str, default = "./dataset", help="Path to dataset directory.")
# Training Info
parser.add_argument(
    "--num-epochs", "-e", type=int, help="Number of epochs.")
parser.add_argument(
    "--resume-training", type=int, default=0, help="Resume training from epoch.")
parser.add_argument(
    "--batch-size", type=int, default=8)
parser.add_argument(
    "--num-frames-sequence", type=int, default=9)
# Image Info
parser.add_argument(
    "--original-image-shape", default=(1920,1080))
parser.add_argument(
    "--processed-image-shape", default=(320,128))
configs = parser.parse_args()

