#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Internal ESM feature module.

"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
import esm
from Bio.PDB import PDBParser


AA20 = [
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y",
]

AA_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def parse_mut(mut):
    m = re.fullmatch(
        r"([A-Z])(\d+)([A-Z])",
        mut.strip().upper(),
    )
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut}")
    return m.group(1), int(m.group(2)), m.group(3)


def extract_sequence_and_mapping(
    pdb_path: str | Path,
    chain_id: str,
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())

    if chain_id not in model:
        raise KeyError(
            f"Chain {chain_id} not found in {pdb_path}"
        )

    chain = model[chain_id]
    seq = []
    mapping = {}
    idx = 0

    for res in chain:
        het, resseq, icode = res.id
        if het != " " or icode.strip():
            continue

        resname = res.resname.strip().upper()
        if resname not in AA_THREE_TO_ONE:
            continue

        aa = AA_THREE_TO_ONE[resname]
        seq.append(aa)

        if resseq not in mapping:
            mapping[resseq] = idx

        idx += 1

    return "".join(seq), mapping


def compute_hi(
    model,
    alphabet,
    seq: str,
    idx0: int,
    device,
) -> float:
    seq_mask = list(seq)
    seq_mask[idx0] = "<mask>"
    seq_mask = "".join(seq_mask)

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter(
        [("protein", seq_mask)]
    )
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens)
        logits = out["logits"][0, idx0 + 1]

    probs = torch.softmax(logits, dim=-1)
    aa_ids = [alphabet.get_idx(a) for a in AA20]
    p = probs[aa_ids].cpu().numpy()

    eps = 1e-12
    hi = -np.sum(p * np.log2(p + eps))
    return float(hi)


def compute_esm_feature(
    pdb_id: str,
    chain: str,
    mut: str,
    script_dir: Path | None = None,
) -> float:
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path(script_dir).resolve()

    pdb_path = script_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(str(pdb_path))

    _, pos, _ = parse_mut(mut)

    model, alphabet = (
        esm.pretrained.esm2_t33_650M_UR50D()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    seq, mapping = extract_sequence_and_mapping(
        pdb_path,
        chain,
    )

    if pos not in mapping:
        raise KeyError(
            f"Residue {pos} not found in chain {chain}"
        )

    return compute_hi(
        model,
        alphabet,
        seq,
        mapping[pos],
        device,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Internal Pred-MutPRI ESM component."
    )
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    ap.parse_args()

    raise SystemExit(
        "run_esm.py is an internal component. "
        "Use ./predict.sh for prediction."
    )


if __name__ == "__main__":
    main()
