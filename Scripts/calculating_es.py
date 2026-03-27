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

PROTEIN_CHAINS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
ALIGN_BACKBONE_ATOMS = ("N", "CA", "C")


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


def get_chain_residue_map(struct, chain_id):
    model = list(struct)[0]
    if chain_id not in model:
        return {}
    residue_map = {}
    for res in model[chain_id]:
        if res.id[0] != " ":
            continue
        residue_map[res.id[1]] = res
    return residue_map


def collect_backbone_atom_pairs(struct_wt, struct_mut, chain_id):
    wt_res_map = get_chain_residue_map(struct_wt, chain_id)
    mut_res_map = get_chain_residue_map(struct_mut, chain_id)

    common_resseqs = sorted(set(wt_res_map) & set(mut_res_map))
    wt_coords = []
    mut_coords = []

    for resseq in common_resseqs:
        wt_res = wt_res_map[resseq]
        mut_res = mut_res_map[resseq]
        for atom_name in ALIGN_BACKBONE_ATOMS:
            if atom_name in wt_res and atom_name in mut_res:
                wt_coords.append(wt_res[atom_name].get_coord())
                mut_coords.append(mut_res[atom_name].get_coord())

    if not wt_coords:
        raise ValueError(
            f"No common backbone N/CA/C atoms found for chain '{chain_id}' in WT and MUT"
        )

    wt_coords = np.asarray(wt_coords, dtype=float)
    mut_coords = np.asarray(mut_coords, dtype=float)

    if wt_coords.shape != mut_coords.shape or wt_coords.shape[0] < 3:
        raise ValueError(
            f"Insufficient backbone anchor atoms for Kabsch alignment on chain '{chain_id}'"
        )
    return wt_coords, mut_coords


def kabsch_fit(P, Q):
    """
    Find rotation R and translation t that best map Q onto P:
        Q_aligned = Q @ R + t
    using the Kabsch algorithm.
    P, Q: (N, 3)
    """
    if P.shape != Q.shape:
        raise ValueError("Kabsch input shapes do not match")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Kabsch input must be of shape (N, 3)")
    if P.shape[0] < 3:
        raise ValueError("At least 3 points are required for Kabsch alignment")

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = centroid_P - centroid_Q @ R
    return R, t


def apply_transform_to_structure(struct, R, t):
    for atom in struct.get_atoms():
        coord = atom.get_coord()
        atom.set_coord(coord @ R + t)


def compute_chain_map_and_superimpose(struct_wt, struct_mut, mutated_chain):
    model_wt = list(struct_wt)[0]
    model_mut = list(struct_mut)[0]

    wt_chains = [c.id for c in model_wt if c.id in PROTEIN_CHAINS]
    mut_chains = [c.id for c in model_mut if c.id in PROTEIN_CHAINS]

    if mutated_chain not in wt_chains:
        raise ValueError(f"WT has no chain '{mutated_chain}'")
    if mutated_chain not in mut_chains:
        raise ValueError(f"MUT has no chain '{mutated_chain}'")

    anchor_wt, anchor_mut = collect_backbone_atom_pairs(struct_wt, struct_mut, mutated_chain)
    R, t = kabsch_fit(anchor_wt, anchor_mut)
    apply_transform_to_structure(struct_mut, R, t)

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
