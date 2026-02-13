#this script takes in a .txt file contianing photon arrival times
#and creates a numpy array (.npz file) containing the bins of the photon counting histogram as well as metadata

#assumes mt clock is in psec and acquisition time is in sec

"""
example usage:

format: python convert_photons.py "path" --ch num_channels
example: python convert_photons.py "C:\data\data.txt" --ch 2

data = np.load("path.npz", allow_pickle=False)
header_json = data["header_json"].item()
ch1 = data["Ch_1"]                      
ch3 = data["Ch_3"]                      
print(header_json[:200])
print(ch1[:5], ch3[:5])

"""

import argparse
import json
import os
import numpy as np


#bin edges
pch_edges = np.logspace(np.log10(1), np.log10(8000), 25)
trace_edges = np.linspace(0, 60, int((60)/(500e-6)) + 1)

def parse_header(header_lines):
    header = {"raw_header": "\n".join(header_lines).strip()}

    header["collection_datetime"] = next((s.strip() for s in header_lines if s.strip()), "")

    for line in header_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()

    mt_str = header["MT Clock"]
    mt_val_psec = float(mt_str.split()[0])
    mt_clock_seconds = mt_val_psec * 1e-12

    return header, mt_clock_seconds


def parse_photon_txt(txt_path, channels):
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()


    end_idx = next(i for i, s in enumerate(lines) if "***end header***" in s)
    header_lines = lines[: end_idx + 1]
    data_lines = lines[end_idx + 1 :]

    header, mt_clock_seconds = parse_header(header_lines)

    j = 0
    while j < len(data_lines) and not data_lines[j].strip():
        j += 1
    chan_line = data_lines[j]

    chan_names = [c.strip() for c in chan_line.split("\t") if c.strip()][:channels]

    cols = [[] for _ in range(channels)]
    for s in data_lines[j + 1 :]:
        s = s.strip()
        if not s:
            continue
        toks = s.split()
        for ci in range(channels):
            cols[ci].append(float(toks[ci]) * mt_clock_seconds)

    arrays = [np.asarray(col, dtype=np.float64) for col in cols]
    return header, chan_names, arrays


def save_npz(out_npz_path, source_file, header, chan_names, arrays):
    payload = {
        "header_json": json.dumps(
            {
                "source_file": os.path.abspath(source_file),
                "channels_loaded": chan_names,
                **header,
            },
            indent=2,
        )
    }
    for name, arr in zip(chan_names, arrays):
        histI, _ = np.histogram(arr, trace_edges)
        PCHbins, _ = np.histogram(histI, bins=pch_edges)

        payload[name.replace(" ", "_")] = PCHbins 
    np.savez_compressed(out_npz_path, **payload)


ap = argparse.ArgumentParser(description="Convert photon timestamps .txt -> single .npz (header_json + arrays)")
ap.add_argument("pathToTxt", type=str, help="Path to a .txt file")
ap.add_argument("--ch", type=int, default=1, help="Number of channels to load (first N columns)")
args = ap.parse_args()

in_path = args.pathToTxt
base, _ = os.path.splitext(in_path)
out_path = base + ".npz"

header, chan_names, arrays = parse_photon_txt(in_path, args.ch)
save_npz(out_path, in_path, header, chan_names, arrays)
print(f"wrote {out_path}")

