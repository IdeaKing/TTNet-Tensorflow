import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--work-dir", "-w", type=str, help="Path to directory to save all files.")
parser.add_argument(
    "--data-dir", "-d", type=str, help="Path to dataset directory.")
parser.add_argument(
    "--num-epochs", "-e", type=int, help="Number of epochs.")
parser.add_argument(
    "--resume-training", type=int, default=0, help="Resume training from epoch.")
parser.add_argument(
    "--batch-size", type=int, default=8)
parser.add_argument(
    "--num-frames-sequence", type=int, default=9)
configs = parser.parse_args()

