#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate AlphaFold3 JSON for a single point mutation from a local PDB.

Inputs (same style as you requested):
  -PDB 1URN -CHAIN A -Mut D92A

Assumptions:
- <PDB>.pdb is in the same directory as this script (e.g. ./1URN.pdb)
- JSON will be written to ./af3_json/<PDB>_<CHAIN>_<Mut>.json
- Output dir name is reserved as ./af3_outputs/<PDB>_<CHAIN>_<Mut>/ (not used here unless you later extend)
- This script only generates JSON. (Your bash script will cd into ./alphafold3 and run prediction.)

Notes:
- Uses PDB residue numbering (resseq). Insertion-code residues (e.g. 92A) are skipped to keep mapping simple.
- WT check is enforced: for D92A, the extracted sequence at resseq 92 must be D, otherwise raises.
"""

import argparse
import json
import os
import re
from datetime import datetime
from Bio.PDB import PDBParser
from Bio.Data import IUPACData

# 3-letter AA -> 1-letter
AA3_TO_1 = dict(IUPACData.protein_letters_3to1)
AA3_TO_1_UP = {k.upper(): v for k, v in AA3_TO_1.items()}
AA3_TO_1_UP["MSE"] = "M"

AA20_SET = set("ACDEFGHIKLMNPQRSTVWY")

# nucleic residue names and mapping
NUC_RESNAMES = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU"}
NUC_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "T",
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
}

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)

def parse_mut(mut: str):
    """
    D92A -> (wt='D', pos=92, mt='A')
    """
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut.strip().upper())
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut}. Expected like D92A")
    return m.group(1), int(m.group(2)), m.group(3)

def is_amino(resname: str) -> bool:
    return resname.strip().upper() in AA3_TO_1_UP

def is_nucleic(resname: str) -> bool:
    return resname.strip().upper() in NUC_RESNAMES

def three_to_one(resname: str) -> str:
    return AA3_TO_1_UP.get(resname.strip().upper(), "X")

def determine_chain_type(chain):
    protein_count = 0
    nuc_count = 0

    for res in chain.get_residues():
        het, resseq, icode = res.id
        if het != " ":
            continue
        if str(icode).strip():  # skip insertion codes
            continue

        resname = res.get_resname().strip()
        if is_amino(resname):
            protein_count += 1
        elif is_nucleic(resname):
            nuc_count += 1

    if protein_count > 0 and nuc_count == 0:
        return "protein"
    elif nuc_count > 0 and protein_count == 0:
        return "rna"
    elif protein_count == 0 and nuc_count == 0:
        return "other"
    else:
        return "mixed"

def extract_chain_sequence(chain):
    """
    Return (sequence, res_ids) for standard residues in this chain.
    res_ids aligns with sequence positions and stores PDB resseq numbers.
    """
    seq = []
    res_ids = []

    for res in chain.get_residues():
        het, resseq, icode = res.id
        if het != " ":
            continue
        if str(icode).strip():  # skip insertion codes for stable mapping
            continue

        resname = res.get_resname().strip()

        if is_amino(resname):
            seq.append(three_to_one(resname))
            res_ids.append(resseq)
        elif is_nucleic(resname):
            seq.append(NUC_MAP.get(resname.strip().upper(), "N"))
            res_ids.append(resseq)
        else:
            continue

    return "".join(seq), res_ids

def mutate_seq(seq, res_ids, target_resseq, mutant_aa):
    if target_resseq not in res_ids:
        raise ValueError(
            f"Residue {target_resseq} not found in chain. "
            f"First 30 resseq: {res_ids[:30]} (len={len(res_ids)})"
        )
    idx = res_ids.index(target_resseq)
    lst = list(seq)
    lst[idx] = mutant_aa
    return "".join(lst), idx

def main():
    ap = argparse.ArgumentParser(description="Generate AF3 JSON for a single mutation.")
    ap.add_argument("-PDB", required=True, help="e.g. 1URN (expects ./1URN.pdb)")
    ap.add_argument("-CHAIN", required=True, help="e.g. A")
    ap.add_argument("-Mut", required=True, help="e.g. D92A")
    args = ap.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))
    pdb_path = os.path.join(script_dir, f"{args.PDB}.pdb")
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    wt, pos, mt = parse_mut(args.Mut)
    if mt not in AA20_SET:
        raise ValueError(f"Mutant AA must be one of 20 AA letters, got {mt}")

    label = f"{args.PDB}_{args.CHAIN}_{args.Mut}"

    json_dir = os.path.join(script_dir, "af3_json")
    os.makedirs(json_dir, exist_ok=True)

    log(f"Task label: {label}")
    log(f"PDB: {pdb_path}")
    log(f"Mutation: chain {args.CHAIN} resseq {pos} {wt}->{mt}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(args.PDB, pdb_path)
    model = next(structure.get_models())  # first model only

    protein_entries = []
    rna_entries = []
    found_mut_chain = False

    for chain in model:
        chain_type = determine_chain_type(chain)

        if chain_type == "mixed":
            log(f"⚠ mixed chain skipped: {chain.id}")
            continue
        if chain_type == "other":
            continue

        seq, res_ids = extract_chain_sequence(chain)
        if not seq:
            continue

        if chain_type == "protein":
            if chain.id == args.CHAIN:
                found_mut_chain = True
                mutated, idx0 = mutate_seq(seq, res_ids, pos, mt)

                # WT check
                if seq[idx0] != wt:
                    raise ValueError(
                        f"WT mismatch for {args.Mut} on chain {args.CHAIN}: "
                        f"PDB has {seq[idx0]} at resseq {pos}, but -Mut says WT {wt}. "
                        f"Check numbering / insertion codes / missing residues."
                    )

                protein_entries.append({"protein": {"id": [chain.id], "sequence": mutated}})
                log(f"[OK] Mutated protein chain {chain.id}: resseq {pos} {wt}->{mt} (seq_idx0={idx0})")
            else:
                protein_entries.append({"protein": {"id": [chain.id], "sequence": seq}})

        elif chain_type == "rna":
            rna_entries.append({"rna": {"id": [chain.id], "sequence": seq}})
            log(f"[OK] Added RNA chain {chain.id} (len={len(seq)})")

    if not found_mut_chain:
        raise KeyError(f"Mutation chain {args.CHAIN} not found in first model of {pdb_path}")

    json_data = {
        "name": label,
        "sequences": protein_entries + rna_entries,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }

    json_path = os.path.join(json_dir, f"{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    log(f"[OK] JSON written: {os.path.abspath(json_path)}")

if __name__ == "__main__":
    main()