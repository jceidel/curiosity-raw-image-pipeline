#!/usr/bin/env python3
"""
MSL Mastcam PDS4 EDR Batch Processor
=====================================
Reads PDS4 XML labels and their associated raw IMG files from the
Mars Science Laboratory (Curiosity) Mastcam instrument, applies
Bayer demosaicing, white balance, color correction, and contrast
stretching, then outputs viewable PNG images.

Supports:
  - Single XML label file
  - Multiple XML label files
  - Entire directories (recursively finds all .xml labels)

Usage:
  python process_mastcam_pds.py <file_or_folder> [file_or_folder ...]

Examples:
  python process_mastcam_pds.py 0042ML0002000000E1_DXXX.xml
  python process_mastcam_pds.py label1.xml label2.xml label3.xml
  python process_mastcam_pds.py ./raw_data/
  python process_mastcam_pds.py ./sol_042/ ./sol_043/ extra_label.xml
  python process_mastcam_pds.py ./raw_data/ --output ./processed/
  python process_mastcam_pds.py ./raw_data/ --bayer gbrg
"""

import sys
import os
import glob
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# PDS4 XML Parsing
# ---------------------------------------------------------------------------

def parse_pds4_label(xml_path):
    """
    Parse a PDS4 XML label and extract image metadata.

    Args:
        xml_path: Path to the PDS4 .xml label file.

    Returns:
        dict with keys: file_name, offset, lines, samples, bits, bayer, lut
    """
    ns = {
        "pds": "http://pds.nasa.gov/pds4/pds/v1",
        "img": "http://pds.nasa.gov/pds4/img/v1",
    }
    root = ET.parse(xml_path).getroot()

    # Get referenced data file name
    file_name = root.findtext(".//pds:file_name", namespaces=ns)

    # Find Array_2D_Image element (try namespaced first, then bare)
    arr = root.find(".//pds:Array_2D_Image", ns)
    if arr is None:
        arr = root.find(".//Array_2D_Image")

    if arr is None:
        raise ValueError(f"No Array_2D_Image element found in {xml_path}")

    offset = int(
        arr.findtext(".//pds:offset", namespaces=ns)
        or arr.findtext(".//offset")
    )
    lines = int(
        arr.findtext(
            ".//pds:Axis_Array[pds:axis_name='Line']/pds:elements", namespaces=ns
        )
        or arr.findtext(".//Axis_Array[axis_name='Line']/elements")
    )
    samples = int(
        arr.findtext(
            ".//pds:Axis_Array[pds:axis_name='Sample']/pds:elements", namespaces=ns
        )
        or arr.findtext(".//Axis_Array[axis_name='Sample']/elements")
    )

    info = {
        "file_name": file_name,
        "offset": offset,
        "lines": lines,
        "samples": samples,
        "bits": 8,
        "bayer": "grbg",  # Mastcam Kodak KAI-2020 CCD: GR/BG unit cell
        "lut": None,
    }

    print(f"  📄 XML parsed: offset={offset}, size={samples}×{lines}")
    return info


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_raw_img(path, offset, samples, lines):
    """Read raw Bayer-mosaic image data from a binary IMG file."""
    expected_bytes = samples * lines
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(expected_bytes)

    if len(raw) < expected_bytes:
        raise IOError(
            f"Expected {expected_bytes} bytes but read {len(raw)} "
            f"from {path} at offset {offset}"
        )

    data = np.frombuffer(raw, dtype=np.uint8)
    return data.reshape((lines, samples))


# ---------------------------------------------------------------------------
# Image Processing Pipeline
# ---------------------------------------------------------------------------

def debayer(bayer_img, pattern="grbg"):
    """
    Demosaic a Bayer-pattern image to BGR using OpenCV Variable Number of Gradients (VNG)
    interpolation for maximum quality.

    IMPORTANT — OpenCV Bayer naming convention:
      OpenCV names Bayer patterns from the 2nd row / 2nd column of the CFA,
      NOT from the top-left 2×2 like the rest of the world (MATLAB, camera
      vendors, NASA PDS labels). See: github.com/opencv/opencv/issues/19629

      Real-world (top-left)   →  OpenCV constant needed
      ─────────────────────────────────────────────────
      RGGB                    →  COLOR_BayerBG2BGR_VNG
      GRBG  (Mastcam)         →  COLOR_BayerGB2BGR_VNG
      GBRG                    →  COLOR_BayerGR2BGR_VNG
      BGGR                    →  COLOR_BayerRG2BGR_VNG

    The Mastcam Kodak KAI-2020 CCD uses a GR/BG (GRBG) unit cell as
    documented in Bell et al. 2017 and the NASA Mastcam instrument page:
      "Integrated over each detector is an RGB Bayer pattern filter
       (GR/BG unit cell)."

    Demosaicing algorithms (slowest → fastest):
      EA  — Edge-Aware: can introduce stripe artifacts on low-contrast scenes
      VNG — Variable Number of Gradients: best balance of quality and robustness
      (default bilinear): fastest but leaves visible grid artifacts
    """
    # Map real-world pattern names → OpenCV EA constants (accounting for the
    # one-row/one-column offset in OpenCV's naming convention)
    patterns = {
        "rggb": cv2.COLOR_BayerBG2BGR_VNG,
        "grbg": cv2.COLOR_BayerGB2BGR_VNG,   # Mastcam default
        "gbrg": cv2.COLOR_BayerGR2BGR_VNG,
        "bggr": cv2.COLOR_BayerRG2BGR_VNG,
    }
    code = patterns.get(pattern.lower(), cv2.COLOR_BayerGB2BGR_VNG)
    return cv2.cvtColor(bayer_img, code)


def white_balance(img):
    """Gray-world white balance — normalise R and B channels to the green mean."""
    img = img.astype(np.float32)
    mean_r = np.mean(img[:, :, 2])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])

    if mean_r > 0:
        img[:, :, 2] *= mean_g / mean_r
    if mean_b > 0:
        img[:, :, 0] *= mean_g / mean_b

    return np.clip(img, 0, 255).astype(np.uint8)


def color_correct(img):
    """Reduce blue cast typical of raw Mastcam EDR data."""
    img = img.astype(np.float32)
    img[:, :, 2] *= 1.10  # Slight red boost
    img[:, :, 1] *= 1.00  # Green stays neutral
    img[:, :, 0] *= 0.85  # Reduce blue
    return np.clip(img, 0, 255).astype(np.uint8)


def stretch_contrast(img):
    """Percentile-based contrast stretch (0.5 %–99.5 %)."""
    imin, imax = np.percentile(img, (0.5, 99.5))
    if imax - imin == 0:
        return img
    stretched = np.clip((img - imin) * 255.0 / (imax - imin), 0, 255)
    return stretched.astype(np.uint8)


# ---------------------------------------------------------------------------
# Single-Image Processor
# ---------------------------------------------------------------------------

def process_pds_image(xml_path, output_dir=None, bayer_override=None):
    """
    Process one PDS4 label + IMG pair into a colour PNG.

    Args:
        xml_path:        Path to the .xml label.
        output_dir:      Optional directory for output PNGs (defaults to same
                         directory as the source IMG).
        bayer_override:  Override the default Bayer pattern (rggb/gbrg/grbg/bggr).

    Returns:
        Path to the saved PNG, or None on failure.
    """
    info = parse_pds4_label(xml_path)
    img_path = os.path.join(os.path.dirname(xml_path), info["file_name"])

    if not os.path.isfile(img_path):
        print(f"  ⚠️  IMG file not found: {img_path}")
        return None

    pattern = bayer_override or info["bayer"]

    print(f"  🛰️  Processing: {info['file_name']}")
    print(f"     Offset : {info['offset']} bytes")
    print(f"     Size   : {info['samples']}×{info['lines']}")
    print(f"     Bayer  : {pattern}")

    bayer_img = read_raw_img(img_path, info["offset"], info["samples"], info["lines"])
    rgb = debayer(bayer_img, pattern)
    rgb = white_balance(rgb)
    rgb = color_correct(rgb)
    rgb = stretch_contrast(rgb)

    # Determine output directory
    # If no explicit output_dir, create an "output_png" folder alongside the source
    if output_dir:
        dest_dir = output_dir
    else:
        source_dir = os.path.dirname(xml_path) or "."
        dest_dir = os.path.join(source_dir, "output_png")

    os.makedirs(dest_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0] + "_RGB.png"
    out_path = os.path.join(dest_dir, base_name)

    cv2.imwrite(out_path, rgb)
    print(f"  ✅ Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Batch / Directory Discovery
# ---------------------------------------------------------------------------

def find_xml_labels(path):
    """
    Given a file or directory, return a sorted list of PDS4 XML label paths.

    - If *path* is a file ending in .xml, return it as a one-element list.
    - If *path* is a directory, recursively glob for *.xml files.
    """
    if os.path.isfile(path):
        if path.lower().endswith(".xml"):
            return [path]
        else:
            print(f"⚠️  Skipping non-XML file: {path}")
            return []
    elif os.path.isdir(path):
        labels = sorted(glob.glob(os.path.join(path, "**", "*.xml"), recursive=True))
        return labels
    else:
        print(f"⚠️  Path not found: {path}")
        return []


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-process MSL Mastcam PDS4 EDR images into colour PNGs.",
        epilog=(
            "Examples:\n"
            "  python process_mastcam_pds.py ./raw_data/\n"
            "  python process_mastcam_pds.py label1.xml label2.xml\n"
            "  python process_mastcam_pds.py ./sol_042/ --output ./processed/\n"
            "  python process_mastcam_pds.py ./raw_data/ --bayer gbrg\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more XML label files or directories containing them.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Directory for output PNGs (default: 'output_png/' inside the source directory).",
    )
    parser.add_argument(
        "-b", "--bayer",
        default=None,
        choices=["rggb", "gbrg", "grbg", "bggr"],
        help="Override the Bayer pattern (default: grbg per Mastcam KAI-2020 spec).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Collect all XML labels from every input argument
    all_labels = []
    for input_path in args.inputs:
        all_labels.extend(find_xml_labels(input_path))

    if not all_labels:
        print("❌ No XML label files found. Check your input paths.")
        sys.exit(1)

    print(f"\n🔭 Found {len(all_labels)} XML label(s) to process.\n")

    success = 0
    failed = 0

    for i, xml_path in enumerate(all_labels, 1):
        print(f"[{i}/{len(all_labels)}] {os.path.basename(xml_path)}")
        try:
            result = process_pds_image(
                xml_path,
                output_dir=args.output,
                bayer_override=args.bayer,
            )
            if result:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
        print()

    # Summary
    print("=" * 50)
    print(f"🏁 Batch complete: {success} succeeded, {failed} failed, {len(all_labels)} total.")
    if args.output:
        print(f"   Output directory: {args.output}")
    else:
        print(f"   Output directory: output_png/ (inside each source directory)")


if __name__ == "__main__":
    main()