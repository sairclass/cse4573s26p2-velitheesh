'''
Helper and zip functions.
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''

import os, torch
from pathlib import Path
import zipfile, argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torchvision.transforms import functional as F
import torchvision.io as io
from torchvision.transforms.functional import pil_to_tensor


def is_image_file(path: str | Path) -> bool:
    """
    Quick heuristic: checks extension + file header (magic bytes)
    to ensure it's really an image.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False

    # 1) Fast check by extension
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
        return False

    # 2) Read first few bytes to verify signature
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except Exception:
        return False

    signatures = {
        b"\xFF\xD8": "jpeg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
    }
    if header.startswith(b"RIFF") and b"WEBP" in header[8:16]:
        return True
    return any(header.startswith(sig) for sig in signatures)


def show_image(img: torch.Tensor, delay=1000):
    """Shows an image.
    """
    plt.imshow(F.to_pil_image(img))
    plt.show()

# def read_image(img_path):
#     return io.read_image(img_path, mode=io.ImageReadMode.RGB)

def read_image(img_path, to_rgb=True) -> torch.Tensor:
    """
    Returns CHW uint8 tensor. Uses Pillow instead of torchvision.io.read_image.
    Works for PNG/JPEG/WebP/GIF (and HEIC/AVIF if Pillow plugins installed).
    """
    p = Path(img_path)
    if not is_image_file(p):
        print(f"Skipping non-image file: {p}")
        return None
    
    with Image.open(img_path) as im:
        if to_rgb:
            im = im.convert("RGB")
        t = pil_to_tensor(im)  # CHW, uint8
    return t

def read_images(img_dir):
    res = {}
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path)
        if img is not None:
            res[img_name] = img
    return res

def write_image(input_image: torch.Tensor, output_path: str):
    io.write_png(input_image, output_path)

def bgr_to_rgb(img: torch.Tensor):
    # Convert BGR to RGB by swapping the channels
    return img.flip(dims=[0])[:, :, ::-1]

def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project 2 submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args

def files2zip(files: list, zip_file_name: str, optional_files: list = None):
    optional = set(optional_files or [])
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            path, name = os.path.split(item)

            if not os.path.exists(item):
                if item in optional or name in optional:
                    # optional missing -> skip quietly
                    continue
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)
                continue

            if os.path.isdir(item):
                # add directory contents recursively; keep top-level dir name
                for root, _, filenames in os.walk(item):
                    for fn in filenames:
                        full = os.path.join(root, fn)
                        rel_under_top = os.path.relpath(full, start=item)
                        arcname = os.path.join(name, rel_under_top)
                        zf.write(full, arcname=arcname)
            else:
                # single file
                zf.write(item, arcname=name)


if __name__ == "__main__":
    args = parse_args()
    file_list = ['stitching.py', 'outputs', 'images', 'task2.json', 'bonus1.json', 'bonus2.json']
    files2zip(file_list, 'submission_' + args.ubit + '.zip', optional_files=['bonus1.json', 'bonus2.json'])
