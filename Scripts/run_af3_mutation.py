#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate an AlphaFold3 JSON input for one mutation.

"""

import argparse
import json
import os
import re
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.Data import IUPACData


AA3_TO_1 = dict(IUPACData.protein_letters_3to1)
AA3_TO_1_UP = {
    k.upper(): v
    for k, v in AA3_TO_1.items()
}
AA3_TO_1_UP["MSE"] = "M"

AA20_SET = set("ACDEFGHIKLMNPQRSTVWY")

NUC_RESNAMES = {
    "A", "C", "G", "U", "T",
    "DA", "DC", "DG", "DT", "DU",
}
NUC_MAP = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "T",
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "T",
    "DU": "U",
}


def parse_mut(mut: str):
    m = re.fullmatch(
        r"([A-Z])(\d+)([A-Z])",
        mut.strip().upper(),
    )
    if not m:
        raise ValueError(
            f"Invalid -Mut format: {mut}. Expected like D92A"
        )
    return m.group(1), int(m.group(2)), m.group(3)


def is_amino(resname: str) -> bool:
    return resname.strip().upper() in AA3_TO_1_UP


def is_nucleic(resname: str) -> bool:
    return resname.strip().upper() in NUC_RESNAMES


def three_to_one(resname: str) -> str:
    return AA3_TO_1_UP.get(
        resname.strip().upper(),
        "X",
    )


def determine_chain_type(chain):
    protein_count = 0
    nuc_count = 0

    for res in chain.get_residues():
        het, _, icode = res.id
        if het != " ":
            continue
        if str(icode).strip():
            continue

        resname = res.get_resname().strip()
        if is_amino(resname):
            protein_count += 1
        elif is_nucleic(resname):
            nuc_count += 1

    if protein_count > 0 and nuc_count == 0:
        return "protein"
    if nuc_count > 0 and protein_count == 0:
        return "rna"
    if protein_count == 0 and nuc_count == 0:
        return "other"
    return "mixed"


def extract_chain_sequence(chain):
    seq = []
    res_ids = []

    for res in chain.get_residues():
        het, resseq, icode = res.id
        if het != " ":
            continue
        if str(icode).strip():
            continue

        resname = res.get_resname().strip()

        if is_amino(resname):
            seq.append(three_to_one(resname))
            res_ids.append(resseq)
        elif is_nucleic(resname):
            seq.append(
                NUC_MAP.get(
                    resname.strip().upper(),
                    "N",
                )
            )
            res_ids.append(resseq)

    return "".join(seq), res_ids


def mutate_seq(
    seq: str,
    res_ids,
    target_resseq: int,
    mutant_aa: str,
):
    if target_resseq not in res_ids:
        raise ValueError(
            f"Residue {target_resseq} not found in chain. "
            f"First 30 resseq: {res_ids[:30]} "
            f"(len={len(res_ids)})"
        )

    idx = res_ids.index(target_resseq)
    seq_list = list(seq)
    seq_list[idx] = mutant_aa
    return "".join(seq_list), idx


def generate_af3_json(
    pdb_id: str,
    chain_id: str,
    mut: str,
    pdb_path: Path,
    json_path: Path,
):
    wt, pos, mt = parse_mut(mut)

    if mt not in AA20_SET:
        raise ValueError(
            f"Mutant AA must be one of 20 AA letters, got {mt}"
        )

    label = f"{pdb_id}_{chain_id}_{mut}"

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        pdb_id,
        str(pdb_path),
    )
    model = next(structure.get_models())

    protein_entries = []
    rna_entries = []
    found_mut_chain = False

    for chain in model:
        chain_type = determine_chain_type(chain)

        if chain_type in ("mixed", "other"):
            continue

        seq, res_ids = extract_chain_sequence(chain)
        if not seq:
            continue

        if chain_type == "protein":
            if chain.id == chain_id:
                found_mut_chain = True
                mutated, idx0 = mutate_seq(
                    seq,
                    res_ids,
                    pos,
                    mt,
                )

                if seq[idx0] != wt:
                    raise ValueError(
                        f"WT mismatch for {mut} on chain "
                        f"{chain_id}: PDB has {seq[idx0]} "
                        f"at resseq {pos}, but -Mut says "
                        f"WT {wt}."
                    )

                protein_entries.append({
                    "protein": {
                        "id": [chain.id],
                        "sequence": mutated,
                    }
                })
            else:
                protein_entries.append({
                    "protein": {
                        "id": [chain.id],
                        "sequence": seq,
                    }
                })

        elif chain_type == "rna":
            rna_entries.append({
                "rna": {
                    "id": [chain.id],
                    "sequence": seq,
                }
            })

    if not found_mut_chain:
        raise KeyError(
            f"Mutation chain {chain_id} not found in "
            f"first model of {pdb_path}"
        )

    data = {
        "name": label,
        "sequences": protein_entries + rna_entries,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    ap = argparse.ArgumentParser(
        description="Generate AF3 JSON for one mutation."
    )
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    ap.add_argument(
        "--json_path",
        required=True,
        help="Internal output JSON path.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    pdb_path = script_dir / f"{args.PDB}.pdb"

    if not pdb_path.exists():
        raise FileNotFoundError(
            f"PDB not found: {pdb_path}"
        )

    generate_af3_json(
        args.PDB,
        args.CHAIN,
        args.Mut,
        pdb_path,
        Path(args.json_path).resolve(),
    )


if __name__ == "__main__":
    main()
