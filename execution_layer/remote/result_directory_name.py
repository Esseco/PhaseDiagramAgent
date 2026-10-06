"""Keep a single DFT result associated with its numbered submission."""
from pathlib import Path
import re


def result_directory_name(task_directory, stage):
    directory = Path(task_directory)
    submission = directory.parent.name
    if stage in {"dft_single_point", "dft_relax"} and re.fullmatch(
            r"DFT-(?:single-point|relax)-submission-\d+_remote-\d+", submission):
        return submission
    return directory.name
