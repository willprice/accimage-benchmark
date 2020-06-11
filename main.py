import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import accimage
import PIL.Image
import numpy as np
from typing import List

parser = argparse.ArgumentParser(
    description="Run accimage benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-n", "--n-runs", type=int, default=1000)


@dataclass
class ImgInfo:
    path: Path
    width: int
    height: int


def get_image_info(path):
    img = PIL.Image.open(path)
    return ImgInfo(path, img.width, img.height)


def list_images(root: Path, extensions=("png", "jpg")) -> List[ImgInfo]:
    image_infos = []
    for path in root.iterdir():
        if any([path.suffix.lower().endswith(ext) for ext in extensions]):
            image_infos.append(get_image_info(path))
    return image_infos


def accimage_load(path):
    return accimage.Image(path)


def pil_load(path):
    return PIL.Image.open(path)


def benchmark_op(op_fn, *args, n=100, **kwargs) -> np.ndarray:
    durations = []
    for _ in range(n):
        start = time.time()
        op_fn(*args, **kwargs)
        duration = time.time() - start
        durations.append(duration)
    return np.array(durations)


def report_durations(durations: np.ndarray):
    mean = durations.mean()
    std = durations.std()
    return f"{mean*1e3}ms ± {std*1e3}ms ({len(durations)} runs)"


def crop(img, roi):
    return img.crop(roi)


def resize(img, size, mode):
    return img.resize(size, mode)


def main(args):
    n = args.n_runs
    imgs = list_images(Path("images"))
    for img in imgs:
        path = str(img.path)
        print(f"Profiling results for {img.path.name}")
        print("Loading file:")
        print("pil:      ", report_durations(benchmark_op(pil_load, path, n=n)))
        print("accimage: ", report_durations(benchmark_op(accimage_load, path, n=n)))

        accimage_img = accimage_load(path)
        pil_img = pil_load(path)
        height = pil_img.height
        width = pil_img.width

        print("Cropping:")
        roi = (0, 0, height // 2, width // 2)
        print(
            "accimage: ", report_durations(benchmark_op(crop, accimage_img, roi, n=n))
        )
        print("pil:      ", report_durations(benchmark_op(crop, pil_img, roi, n=n)))

        accimage_img = accimage_load(path)
        pil_img = pil_load(path)
        print("Resizing x2 bilinear:")
        size = (height * 2, width * 2)
        print(
            "accimage: ",
            report_durations(
                benchmark_op(resize, accimage_img, size, PIL.Image.BILINEAR, n=n)
            ),
        )
        print(
            "pil:      ",
            report_durations(
                benchmark_op(resize, pil_img, size, PIL.Image.BILINEAR, n=n)
            ),
        )

        accimage_img = accimage_load(path)
        pil_img = pil_load(path)
        print("Resizing /2 bilinear:")
        size = (height // 2, width // 2)
        print(
                "accimage: ",
                report_durations(
                        benchmark_op(resize, accimage_img, size, PIL.Image.BILINEAR, n=n)
                ),
        )
        print(
                "pil:      ",
                report_durations(
                        benchmark_op(resize, pil_img, size, PIL.Image.BILINEAR, n=n)
                ),
        )



if __name__ == "__main__":
    main(parser.parse_args())
