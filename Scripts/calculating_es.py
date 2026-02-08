#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
import sys
import re
import numpy as np
from Bio import PDB
from Bio.PDB import Superimposer

PROTEIN_CHAINS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

def get_rep_atom(res):
    if "CA" in res:
        return res["CA"]
    for alt in ("C", "N"):
        if alt in res:
            return res[alt]
    return None

def extract_chain_atoms(struct, chain_id):
    model = list(struct)[0]
    if chain_id not in model:
        return []
    entries = []
    for res in model[chain_id]:
        if res.id[0] != " ":
            continue
        atom = get_rep_atom(res)
        if atom is None:
            continue
        entries.append({"resseq": res.id[1], "atom": atom})
    return entries

def compute_chain_map_and_superimpose(struct_wt, struct_mut, mutated_chain):
    model_wt = list(struct_wt)[0]
    model_mut = list(struct_mut)[0]

    wt_chains = [c.id for c in model_wt if c.id in PROTEIN_CHAINS]
    mut_chains = [c.id for c in model_mut if c.id in PROTEIN_CHAINS]

    if mutated_chain not in wt_chains:
        raise ValueError(f"WT has no chain '{mutated_chain}'")
    if mutated_chain not in mut_chains:
        raise ValueError(f"MUT has no chain '{mutated_chain}'")

    e_wt = extract_chain_atoms(struct_wt, mutated_chain)
    e_mut = extract_chain_atoms(struct_mut, mutated_chain)
    L = min(len(e_wt), len(e_mut))
    if L == 0:
        raise ValueError(f"No atoms for chain '{mutated_chain}' in WT or MUT")

    anchor_wt = [x["atom"] for x in e_wt[:L]]
    anchor_mut = [x["atom"] for x in e_mut[:L]]

    sup = Superimposer()
    sup.set_atoms(anchor_wt, anchor_mut)
    sup.apply(list(struct_mut.get_atoms()))

    chain_map = {}
    for cid in wt_chains:
        if cid in mut_chains:
            chain_map[cid] = cid
    return chain_map

def compute_ES(struct_wt, struct_mut, chain_map, mutated_chain, mutated_resi, cutoff):
    wt_chain_atoms = {}
    mut_chain_atoms = {}
    wt_index_map = {}

    for wt_c, mut_c in chain_map.items():
        wt_entries = extract_chain_atoms(struct_wt, wt_c)
        mut_entries = extract_chain_atoms(struct_mut, mut_c)

        wt_chain_atoms[wt_c] = wt_entries
        mut_chain_atoms[mut_c] = mut_entries
        wt_index_map[wt_c] = {entry["resseq"]: idx for idx, entry in enumerate(wt_entries)}

    idx_center = wt_index_map.get(mutated_chain, {}).get(mutated_resi)
    if idx_center is None:
        raise ValueError(f"Mut site not found in WT: chain {mutated_chain}, resseq {mutated_resi}")

    r_i_wt = wt_chain_atoms[mutated_chain][idx_center]["atom"].get_coord()

    wt_all = []
    for cid, entries in wt_chain_atoms.items():
        for idx_in_chain, entry in enumerate(entries):
            wt_all.append({"chain": cid, "idx": idx_in_chain, "atom": entry["atom"]})

    neighbors = []
    for e in wt_all:
        if e["chain"] == mutated_chain and e["idx"] == idx_center:
            continue
        d = np.linalg.norm(e["atom"].get_coord() - r_i_wt)
        if d < cutoff:
            neighbors.append(e)

    mut_target_chain = chain_map[mutated_chain]
    mut_target_entries = mut_chain_atoms[mut_target_chain]
    if idx_center >= len(mut_target_entries):
        return np.nan
    mut_i = mut_target_entries[idx_center]

    ratios = []
    for e in neighbors:
        wt_c = e["chain"]
        idx_j = e["idx"]

        mut_c = chain_map[wt_c]
        mut_entries = mut_chain_atoms[mut_c]
        if idx_j >= len(mut_entries):
            continue

        mut_j = mut_entries[idx_j]

        rij_wt = e["atom"].get_coord() - r_i_wt
        rij_mut = mut_j["atom"].get_coord() - mut_i["atom"].get_coord()

        delta = np.linalg.norm(rij_wt - rij_mut)
        norm = np.linalg.norm(rij_wt)
        if norm < 1e-6:
            continue
        ratios.append(delta / norm)

    return float(np.mean(ratios)) if ratios else np.nan

def sample_index_from_path(p: str):
    m = re.search(r"(?:^|/|\\)seed-1_sample-(\d+)(?:/|\\)", p)
    return int(m.group(1)) if m else None

def find_seed_cif_files(work_dir, label):
    # Your layout: ./af3_outputs/<label>/<label>/seed-1_sample-*/**/*.cif
    base = os.path.join(work_dir, "af3_outputs", label, label)
    if not os.path.isdir(base):
        return []
    files = glob.glob(os.path.join(base, "seed-1_sample-*", "**", "*.cif"), recursive=True)
    picked = []
    for f in files:
        k = sample_index_from_path(f)
        if k is not None:
            picked.append((k, f))
    picked.sort(key=lambda x: x[0])
    return picked

def safe_rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(description="Compute Avg_ES from AF3 seed outputs, then cleanup.")
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    ap.add_argument("--cutoff", type=float, default=13.0)
    ap.add_argument("--no_cleanup", action="store_true")
    args = ap.parse_args()

    work_dir = os.getcwd()
    wt_pdb = os.path.join(work_dir, f"{args.PDB}.pdb")
    if not os.path.exists(wt_pdb):
        print(f"[ERROR] WT PDB not found: {wt_pdb}", file=sys.stderr)
        sys.exit(1)

    m = re.fullmatch(r"[A-Za-z](\d+)[A-Za-z]", args.Mut.strip())
    if not m:
        print(f"[ERROR] -Mut must look like D92A, got {args.Mut}", file=sys.stderr)
        sys.exit(1)
    mutated_resi = int(m.group(1))

    label = f"{args.PDB}_{args.CHAIN}_{args.Mut}"
    seed_cifs = find_seed_cif_files(work_dir, label)
    if not seed_cifs:
        base = os.path.join(work_dir, "af3_outputs", label, label)
        print(f"[ERROR] No seed .cif found under: {base}/seed-1_sample-*/", file=sys.stderr)
        sys.exit(1)

    parser_pdb = PDB.PDBParser(QUIET=True)
    parser_cif = PDB.MMCIFParser(QUIET=True)
    struct_wt = parser_pdb.get_structure(f"WT_{args.PDB}", wt_pdb)

    es_values = []
    for sample_idx, cif in seed_cifs:
        try:
            struct_mut = parser_cif.get_structure(f"MUT_{label}_s{sample_idx}", cif)
            chain_map = compute_chain_map_and_superimpose(struct_wt, struct_mut, args.CHAIN)
            es = compute_ES(struct_wt, struct_mut, chain_map, args.CHAIN, mutated_resi, args.cutoff)
        except Exception:
            es = np.nan

        if not np.isnan(es):
            es_values.append(es)

    avg_es = float(np.mean(es_values)) if es_values else np.nan
    # 只打印平均 ES
    if np.isnan(avg_es):
        print("NaN")
    else:
        print(f"{avg_es:.6f}")

    if not args.no_cleanup:
        safe_rmtree(os.path.join(work_dir, "af3_json"))
        safe_rmtree(os.path.join(work_dir, "logs"))
        safe_rmtree(os.path.join(work_dir, "af3_outputs"))

if __name__ == "__main__":
    main()