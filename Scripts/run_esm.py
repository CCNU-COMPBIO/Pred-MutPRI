#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import numpy as np
import torch
import esm
from Bio.PDB import PDBParser

AA20 = ["A","C","D","E","F","G","H","I","K","L",
        "M","N","P","Q","R","S","T","V","W","Y"]

AA_THREE_TO_ONE = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}

def parse_mut(mut):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut.strip().upper())
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut}")
    return m.group(1), int(m.group(2)), m.group(3)

def extract_sequence_and_mapping(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())

    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} not found in {pdb_path}")

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

def compute_hi(model, alphabet, seq, idx0, device):
    seq_mask = list(seq)
    seq_mask[idx0] = "<mask>"
    seq_mask = "".join(seq_mask)

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", seq_mask)])
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    args = ap.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))
    pdb_path = os.path.join(script_dir, f"{args.PDB}.pdb")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(pdb_path)

    wt, pos, mut = parse_mut(args.Mut)

    print("[INFO] Loading ESM-2 model esm2_t33_650M_UR50D ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    seq, mapping = extract_sequence_and_mapping(pdb_path, args.CHAIN)

    if pos not in mapping:
        raise KeyError(f"Residue {pos} not found in chain {args.CHAIN}")

    idx0 = mapping[pos]

    hi = compute_hi(model, alphabet, seq, idx0, device)

    print(f"{hi:.6f}")

if __name__ == "__main__":
    main()
