# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:03:21 2026

@author: xuwan
"""

# -*- coding: utf-8 -*-
"""
PRISM-RNA Network Features (All-in-One)

Pipeline:
  Step1) Build 4 interaction tables (AAinter_w/m, NAinter_w/m) from WT and mutant PDB
  Step2) Cytoscape Analyze Network for 4 CSVs (export node/edge tables)
  Step3) Compute node/edge features + Δ features -> <PDB>_network_features.csv
  Step4) Print selected features in required order, then cleanup intermediate files

Usage:
  python PRISM_network_allinone.py -PDB 1URN -CHAIN A -Mut D92A

Requirements:
  - Biopython, numpy, pandas, py4cytoscape
  - Cytoscape GUI running (Cytoscape Automation enabled)
  - WT PDB:   ./<PDB>.pdb
  - Mut PDB:  ./<PDB>_1.pdb   (FoldX BuildModel output; must exist)
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Cytoscape (Step2)
# -------------------------
import py4cytoscape as p4c

# -------------------------
# Biopython (Step1)
# -------------------------
from Bio.PDB import PDBParser

# =========================
# Step4: required output order
# =========================
FEATURE_ORDER = [
    "AverageShortestPathLength_NAinter_w",
    "Δ_NumberOfUndirectedEdges_NAinter_noweight",
    "NeighborhoodConnectivity_AAinter_w",
    "BetweennessCentrality_AAinter_m",
    "NeighborhoodConnectivity_AAinter_m",
    "Other_Ratio_AAinter_w",
    "N-O/C-O/C-N_Count_AAinter_m",
    "N-O/C-O/C-N_Mean_AAinter_m",
    "Other_Ratio_AAinter_m",
    "Δ_H-O/H-N_Count_AAinter",
    "Total_interactions_NAinter_w",
    "N-O/C-O/C-N_Count_NAinter_w",
    "Other_Mean_NAinter_w",
    "N-O/C-O/C-N_Ratio_NAinter_w",
    "Other_Ratio_NAinter_w",
    "Evaluation_NAinter_w",
    "Total_interactions_NAinter_m",
    "N-O/C-O/C-N_Count_NAinter_m",
    "N-O/C-O/C-N_Ratio_NAinter_m",
    "Other_Ratio_NAinter_m",
    "Evaluation_NAinter_m",
    "Δ_Total_interactions_NAinter",
    "Δ_H-O/H-N_Count_NAinter",
    "Δ_N-O/C-O/C-N_Count_NAinter",
    "Δ_N-O/C-O/C-N_Mean_NAinter",
    "Δ_Other_Mean_NAinter",
    "Δ_H-O/H-N_Ratio_NAinter",
    "Δ_N-O/C-O/C-N_Ratio_NAinter",
    "Δ_Other_Ratio_NAinter",
    "Δ_Evaluation_NAinter",
]

# =========================
# Step3: node features list (Cytoscape Analyze Network node table)
# =========================
NODE_FEATURES = [
    "AverageShortestPathLength", "ClusteringCoefficient", "ClosenessCentrality",
    "PartnerOfMultiEdgedNodePairs", "SelfLoops", "Eccentricity",
    "Stress", "Degree", "BetweennessCentrality", "NeighborhoodConnectivity",
    "NumberOfDirectedEdges", "NumberOfUndirectedEdges", "Radiality", "TopologicalCoefficient"
]

# =========================
# Helpers
# =========================
def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s).strip().strip(".")
    return s[:180] if len(s) > 180 else s


def parse_mut(mut: str) -> Tuple[str, int, str]:
    """
    D92A -> (D, 92, A)
    """
    mut = mut.strip()
    m = re.fullmatch(r"([A-Za-z])(\d+)([A-Za-z])", mut)
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut} (expected like D92A)")
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


# =========================
# Step1: build interaction tables
# =========================
PARSER = PDBParser(QUIET=True)

AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
}
NT_NAMES = {"A","U","G","C","DA","DT","DG","DC"}  # tolerate DNA labels too


def get_structure(pdb_path: Path):
    return PARSER.get_structure(pdb_path.stem, str(pdb_path))


def is_standard_residue(res) -> bool:
    return (res.get_resname().strip().upper() in AA3) and (res.id[0] == " ")


def is_nucleotide(res) -> bool:
    return res.get_resname().strip().upper() in NT_NAMES


def residue_uid(chain_id: str, res) -> str:
    """
    Label residue like: LYS_A21
    """
    resname = res.get_resname().strip().upper()
    resseq = int(res.id[1])
    return f"{resname}_{chain_id}{resseq}"


def atoms_of_residue(res):
    # ignore H if present
    out = []
    for a in res.get_atoms():
        try:
            if a.element == "H":
                continue
        except Exception:
            pass
        out.append(a)
    return out


def find_residue(structure, chain_id: str, pos: int):
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in structure {structure.id}")
    chain = model[chain_id]
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if int(res.id[1]) == pos:
            return chain, res
    raise ValueError(f"Residue position {pos} not found in chain {chain_id}")


def add_source_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Adds 'source' and 'target' as LAST TWO columns:
      AA: source = Target_Residue + "_" + Target_Atom
          target = Neighbor_Residue + "_" + Neighbor_Atom
      NA: source = Protein_Residue + "_" + Protein_Atom
          target = Nucleic_Residue + "_" + Nucleic_Atom
    """
    df = df.copy()
    if kind == "AA":
        c1, c2, c3, c4 = "Target_Residue", "Target_Atom", "Neighbor_Residue", "Neighbor_Atom"
    elif kind == "NA":
        c1, c2, c3, c4 = "Protein_Residue", "Protein_Atom", "Nucleic_Residue", "Nucleic_Atom"
    else:
        raise ValueError("kind must be 'AA' or 'NA'")

    if df.empty:
        df["source"] = pd.Series(dtype=str)
        df["target"] = pd.Series(dtype=str)
    else:
        df["source"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        df["target"] = df[c3].astype(str) + "_" + df[c4].astype(str)

    base_cols = [c for c in df.columns if c not in ["source", "target"]]
    return df[base_cols + ["source", "target"]]


def extract_aa_aa_contacts(pdb_path: Path, chain_id: str, mut_pos: int, cutoff: float = 5.0) -> pd.DataFrame:
    """
    Atom-level AA-AA contacts between mutated residue and any protein residue (all chains) within cutoff.
    """
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    if not mut_atoms:
        return pd.DataFrame(columns=["Target_Residue","Target_Atom","Neighbor_Residue","Neighbor_Atom","Distance"])

    rows = []
    mut_uid = residue_uid(chain_id, mut_res)
    model = next(structure.get_models())

    for ch in model.get_chains():
        ch_id = ch.id
        for res in ch.get_residues():
            if not is_standard_residue(res):
                continue
            if is_nucleotide(res):
                continue
            # skip itself
            if ch_id == chain_id and res.id == mut_res.id:
                continue

            neigh_atoms = atoms_of_residue(res)
            if not neigh_atoms:
                continue
            neigh_uid = residue_uid(ch_id, res)

            for a in mut_atoms:
                a_coord = a.coord
                for b in neigh_atoms:
                    d = float(np.linalg.norm(a_coord - b.coord))
                    if d <= cutoff:
                        rows.append({
                            "Target_Residue": mut_uid,
                            "Target_Atom": a.get_name().strip(),
                            "Neighbor_Residue": neigh_uid,
                            "Neighbor_Atom": b.get_name().strip(),
                            "Distance": d,
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Target_Residue","Target_Atom","Neighbor_Residue","Neighbor_Atom","Distance"])
    df = add_source_target(df, kind="AA")
    return df


def extract_aa_na_contacts(pdb_path: Path, chain_id: str, mut_pos: int, cutoff: float = 10.0) -> pd.DataFrame:
    """
    Atom-level AA-NA contacts between mutated residue and nucleotides (all chains) within cutoff.
    """
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    if not mut_atoms:
        return pd.DataFrame(columns=["Protein_Residue","Protein_Atom","Nucleic_Residue","Nucleic_Atom","Distance"])

    rows = []
    prot_uid = residue_uid(chain_id, mut_res)
    model = next(structure.get_models())

    for ch in model.get_chains():
        ch_id = ch.id
        for res in ch.get_residues():
            if not is_nucleotide(res):
                continue

            nt_atoms = atoms_of_residue(res)
            if not nt_atoms:
                continue
            nt_uid = residue_uid(ch_id, res)  # reuse residue_uid for nucleotides too

            for a in mut_atoms:
                a_coord = a.coord
                for b in nt_atoms:
                    d = float(np.linalg.norm(a_coord - b.coord))
                    if d <= cutoff:
                        rows.append({
                            "Protein_Residue": prot_uid,
                            "Protein_Atom": a.get_name().strip(),
                            "Nucleic_Residue": nt_uid,
                            "Nucleic_Atom": b.get_name().strip(),
                            "Distance": d,
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Protein_Residue","Protein_Atom","Nucleic_Residue","Nucleic_Atom","Distance"])
    df = add_source_target(df, kind="NA")
    return df


def step1_make_interaction_tables(script_dir: Path, pdb_id: str, chain: str, mut: str) -> Dict[str, Path]:
    """
    Creates:
      <PDB>_AAinter_w.csv, <PDB>_AAinter_m.csv, <PDB>_NAinter_w.csv, <PDB>_NAinter_m.csv
    Returns dict suffix->csv_path
    """
    _, mut_pos, _ = parse_mut(mut)

    wt_pdb = script_dir / f"{pdb_id}.pdb"
    mut_pdb = script_dir / f"{pdb_id}_1.pdb"
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB not found: {wt_pdb}")
    if not mut_pdb.exists():
        raise FileNotFoundError(f"Mutant PDB not found: {mut_pdb} (expect FoldX output <PDB>_1.pdb)")

    outs = {}

    # WT
    aa_w = extract_aa_aa_contacts(wt_pdb, chain, mut_pos, cutoff=5.0)
    na_w = extract_aa_na_contacts(wt_pdb, chain, mut_pos, cutoff=10.0)

    # Mut
    aa_m = extract_aa_aa_contacts(mut_pdb, chain, mut_pos, cutoff=5.0)
    na_m = extract_aa_na_contacts(mut_pdb, chain, mut_pos, cutoff=10.0)

    out_aa_w = script_dir / f"{pdb_id}_AAinter_w.csv"
    out_aa_m = script_dir / f"{pdb_id}_AAinter_m.csv"
    out_na_w = script_dir / f"{pdb_id}_NAinter_w.csv"
    out_na_m = script_dir / f"{pdb_id}_NAinter_m.csv"

    aa_w.to_csv(out_aa_w, index=False, encoding="utf-8-sig")
    aa_m.to_csv(out_aa_m, index=False, encoding="utf-8-sig")
    na_w.to_csv(out_na_w, index=False, encoding="utf-8-sig")
    na_m.to_csv(out_na_m, index=False, encoding="utf-8-sig")

    outs["AAinter_w"] = out_aa_w
    outs["AAinter_m"] = out_aa_m
    outs["NAinter_w"] = out_na_w
    outs["NAinter_m"] = out_na_m

    return outs


# =========================
# Step2: Cytoscape analyze networks, export node/edge tables
# =========================
def read_edge_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "source" in df.columns and "target" in df.columns:
        edges = df[["source", "target"]].copy()
    else:
        if df.shape[1] < 2:
            raise ValueError(f"CSV列数不足，无法构建边表: {csv_path}")
        edges = pd.DataFrame({
            "source": df.iloc[:, -2].astype(str),
            "target": df.iloc[:, -1].astype(str),
        })

    edges = edges.dropna()
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges = edges[(edges["source"] != "") & (edges["target"] != "")]
    edges = edges.reset_index(drop=True)
    if edges.empty:
        raise ValueError(f"边表为空: {csv_path}")
    return edges


def create_and_analyze_network(edges: pd.DataFrame, net_name: str, directed: bool = False) -> int:
    try:
        net_suid = p4c.networks.create_network_from_data_frames(edges=edges, title=net_name)
    except TypeError:
        net_suid = p4c.networks.create_network_from_data_frames(edges=edges)
        try:
            p4c.networks.rename_network(net_name, net_suid)
        except Exception:
            alt = f"{net_name}_{net_suid}"
            p4c.networks.rename_network(alt, net_suid)
            net_name = alt

    try:
        p4c.networks.set_current_network(net_suid)
    except Exception:
        pass

    try:
        p4c.layout.apply_layout("force-directed", network=net_suid)
    except Exception:
        pass

    try:
        p4c.analyze_network(network=net_suid, directed=directed)
    except TypeError:
        p4c.networks.set_current_network(net_suid)
        p4c.analyze_network()

    return net_suid


def export_tables(net_suid: int, out_prefix: Path) -> Tuple[Path, Path]:
    try:
        node_table = p4c.tables.get_table_columns("node", network=net_suid)
        edge_table = p4c.tables.get_table_columns("edge", network=net_suid)
    except TypeError:
        node_table = p4c.tables.get_table_columns("node")
        edge_table = p4c.tables.get_table_columns("edge")

    node_out = Path(str(out_prefix) + "_nodes.csv")
    edge_out = Path(str(out_prefix) + "_edges.csv")

    pd.DataFrame(node_table).to_csv(node_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(edge_table).to_csv(edge_out, index=False, encoding="utf-8-sig")
    return node_out, edge_out


def step2_cytoscape_analyze(script_dir: Path, pdb_id: str):
    p4c.cytoscape_ping()

    out_dir = script_dir / "cytoscape_results" / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = [
        (script_dir / f"{pdb_id}_AAinter_w.csv", f"{pdb_id}_AAinter_w"),
        (script_dir / f"{pdb_id}_AAinter_m.csv", f"{pdb_id}_AAinter_m"),
        (script_dir / f"{pdb_id}_NAinter_w.csv", f"{pdb_id}_NAinter_w"),
        (script_dir / f"{pdb_id}_NAinter_m.csv", f"{pdb_id}_NAinter_m"),
    ]

    error_log = []
    for csv_path, net_base in input_files:
        if not csv_path.exists():
            error_log.append((csv_path.name, "File not found"))
            continue

        net_name = safe_filename(net_base)

        try:
            edges = read_edge_table(csv_path)
            net_suid = create_and_analyze_network(edges, net_name=net_name, directed=False)
            out_prefix = out_dir / net_name
            export_tables(net_suid, out_prefix)

            try:
                p4c.networks.delete_network(net_suid)
            except Exception:
                pass

        except Exception as e:
            error_log.append((csv_path.name, str(e)))

    if error_log:
        pd.DataFrame(error_log, columns=["file", "error"]).to_csv(out_dir / "error_log.csv", index=False, encoding="utf-8-sig")


# =========================
# Step3: compute network features
# =========================
def classify_node_type(name: str) -> str:
    """
    Node 'name' example:
      LYS_A21_CB -> amino (prefix len==3)
      G_A21_P    -> nuc   (prefix len<=2)
    """
    try:
        prefix = str(name).split("_")[0].strip()
        if len(prefix) <= 2:
            return "nuc"
        elif len(prefix) == 3:
            return "amino"
        return "unknown"
    except Exception:
        return "unknown"


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


def summarize_nodes(nodes_csv: Path, network_kind: str, suffix: str) -> Dict[str, float]:
    """
    network_kind: "AA" or "NA"
    weighting (fixed then output):
      - NA: weighted = 0.6 * aa_mean + 0.4 * na_mean
      - AA: weighted = 1.0 * aa_mean + 0.0 * na_mean
    """
    out: Dict[str, float] = {}

    # init
    for feat in NODE_FEATURES:
        out[f"{feat}_{suffix}"] = 0.0
    out[f"n_amino_{suffix}"] = 0.0
    out[f"n_nuc_{suffix}"] = 0.0
    out[f"n_unknown_{suffix}"] = 0.0

    if not nodes_csv.exists():
        return out

    df = pd.read_csv(nodes_csv)
    if "name" not in df.columns:
        raise ValueError(f"[{nodes_csv.name}] missing required column 'name'.")

    for feat in NODE_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    df["__type__"] = df["name"].apply(classify_node_type)

    aa_df = df[df["__type__"] == "amino"]
    na_df = df[df["__type__"] == "nuc"]
    unk_df = df[df["__type__"] == "unknown"]

    out[f"n_amino_{suffix}"] = float(len(aa_df))
    out[f"n_nuc_{suffix}"] = float(len(na_df))
    out[f"n_unknown_{suffix}"] = float(len(unk_df))

    # fixed weights
    if network_kind.upper() == "NA":
        w_aa, w_na = 0.6, 0.4
    else:  # AA
        w_aa, w_na = 1.0, 0.0

    for feat in NODE_FEATURES:
        aa_mean = safe_mean(aa_df[feat]) if len(aa_df) else 0.0
        na_mean = safe_mean(na_df[feat]) if len(na_df) else 0.0
        out[f"{feat}_{suffix}"] = float(w_aa * aa_mean + w_na * na_mean)

    return out


def extract_element(atom_name: str) -> str:
    atom_name = str(atom_name).strip()
    if "_" in atom_name:
        last = atom_name.split("_")[-1]
        if len(last) > 0:
            return last[0].upper()
    return "Unrecognized"


def classify_interaction(source: str, target: str) -> str:
    """
    Use unordered set so H-O == O-H (same for all pairs)
    Groups:
      - H–O/H–N
      - N–O/C–O/C–N
      - Other
      - Unrecognized
    """
    src = extract_element(source)
    tgt = extract_element(target)
    if "Unrecognized" in (src, tgt):
        return "Unrecognized"

    pair = {src, tgt}
    if pair in [{"H","O"}, {"H","N"}]:
        return "H–O/H–N"
    if pair in [{"N","O"}, {"C","O"}, {"C","N"}]:
        return "N–O/C–O/C–N"
    return "Other"


def summarize_edges(edges_csv: Path, suffix: str) -> Dict[str, float]:
    """
    Uses EdgeBetweenness.
    Weighted evaluation:
      Evaluation = 0.6*mean(H–O/H–N) + 0.3*mean(N–O/C–O/C–N) + 0.1*mean(Other)

    NOTE: per request, names do NOT start with 'Edge'
    """
    out: Dict[str, float] = {}
    base_names = [
        "Total_interactions",
        "H-O/H-N_Count", "N-O/C-O/C-N_Count", "Other_Count", "Unrec_Count",
        "H-O/H-N_Mean", "N-O/C-O/C-N_Mean", "Other_Mean",
        "H-O/H-N_Ratio", "N-O/C-O/C-N_Ratio", "Other_Ratio",
        "Evaluation",
    ]
    for bn in base_names:
        out[f"{bn}_{suffix}"] = 0.0

    if not edges_csv.exists():
        return out

    df = pd.read_csv(edges_csv)
    required = {"source", "target", "EdgeBetweenness"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[{edges_csv.name}] missing required columns: {missing}")

    df["EdgeBetweenness"] = pd.to_numeric(df["EdgeBetweenness"], errors="coerce")
    df["__type__"] = df.apply(lambda r: classify_interaction(r["source"], r["target"]), axis=1)

    total = len(df)
    counts = df["__type__"].value_counts().to_dict()

    def mean_for(t: str) -> float:
        x = df.loc[df["__type__"] == t, "EdgeBetweenness"].dropna()
        return float(x.mean()) if len(x) else 0.0

    mean_h = mean_for("H–O/H–N")
    mean_no = mean_for("N–O/C–O/C–N")
    mean_other = mean_for("Other")

    hon = counts.get("H–O/H–N", 0)
    nocon = counts.get("N–O/C–O/C–N", 0)
    oth = counts.get("Other", 0)
    unrec = counts.get("Unrecognized", 0)

    out[f"Total_interactions_{suffix}"] = float(total)
    out[f"H-O/H-N_Count_{suffix}"] = float(hon)
    out[f"N-O/C-O/C-N_Count_{suffix}"] = float(nocon)
    out[f"Other_Count_{suffix}"] = float(oth)
    out[f"Unrec_Count_{suffix}"] = float(unrec)

    out[f"H-O/H-N_Mean_{suffix}"] = float(mean_h)
    out[f"N-O/C-O/C-N_Mean_{suffix}"] = float(mean_no)
    out[f"Other_Mean_{suffix}"] = float(mean_other)

    # ratios (exclude unrecognized from denominator? 你原脚本用 total 做分母；这里保持 total)
    denom = float(total) if total else 1.0
    out[f"H-O/H-N_Ratio_{suffix}"] = float(hon / denom)
    out[f"N-O/C-O/C-N_Ratio_{suffix}"] = float(nocon / denom)
    out[f"Other_Ratio_{suffix}"] = float(oth / denom)

    value = 0.6 * mean_h + 0.3 * mean_no + 0.1 * mean_other
    out[f"Evaluation_{suffix}"] = float(value)

    return out


def add_edge_deltas(features: Dict[str, float], net: str) -> None:
    """
    Δ_<Feature>_<net> = <Feature>_<net>_m - <Feature>_<net>_w
    (edge only: Total/H-O/H-N/.../Evaluation)
    """
    edge_bases = [
        "Total_interactions",
        "H-O/H-N_Count", "N-O/C-O/C-N_Count", "Other_Count", "Unrec_Count",
        "H-O/H-N_Mean", "N-O/C-O/C-N_Mean", "Other_Mean",
        "H-O/H-N_Ratio", "N-O/C-O/C-N_Ratio", "Other_Ratio",
        "Evaluation",
    ]
    for bn in edge_bases:
        k_m = f"{bn}_{net}_m"
        k_w = f"{bn}_{net}_w"
        v_m = float(features.get(k_m, 0.0))
        v_w = float(features.get(k_w, 0.0))
        features[f"Δ_{bn}_{net}"] = v_m - v_w


def add_node_deltas(features: Dict[str, float], net: str) -> None:
    """
    Node features are already 'fixed-weighted' (AA or NA rule) inside summarize_nodes().
    Δ_<NodeFeat>_<net> = <NodeFeat>_<net>_m - <NodeFeat>_<net>_w
    """
    for feat in NODE_FEATURES:
        k_m = f"{feat}_{net}_m"
        k_w = f"{feat}_{net}_w"
        v_m = float(features.get(k_m, 0.0))
        v_w = float(features.get(k_w, 0.0))
        features[f"Δ_{feat}_{net}"] = v_m - v_w


def compute_delta_NumberOfUndirectedEdges_NA_amino_only(base_dir: Path, pdb_id: str) -> float:
    """
    Special requested feature:
      Δ_NumberOfUndirectedEdges_NAinter_noweight
    Definition:
      (mean NumberOfUndirectedEdges over AMINO nodes in NAinter_m)
      - (mean NumberOfUndirectedEdges over AMINO nodes in NAinter_w)
    No nucleotide included; no weighting.
    """
    w_csv = base_dir / f"{pdb_id}_NAinter_w_nodes.csv"
    m_csv = base_dir / f"{pdb_id}_NAinter_m_nodes.csv"

    def aa_mean(csvp: Path) -> float:
        if not csvp.exists():
            return 0.0
        df = pd.read_csv(csvp)
        if "name" not in df.columns:
            return 0.0
        if "NumberOfUndirectedEdges" not in df.columns:
            return 0.0
        df["__type__"] = df["name"].apply(classify_node_type)
        aa = df[df["__type__"] == "amino"]
        return safe_mean(aa["NumberOfUndirectedEdges"]) if len(aa) else 0.0

    return float(aa_mean(m_csv) - aa_mean(w_csv))


def step3_compute_network_features(script_dir: Path, pdb_id: str, chain: str, mut: str) -> Path:
    base_dir = script_dir / "cytoscape_results" / pdb_id
    base_dir.mkdir(parents=True, exist_ok=True)

    suffixes = ["AAinter_w", "AAinter_m", "NAinter_w", "NAinter_m"]

    features_all: Dict[str, float] = {
        "PDB": pdb_id,
        "CHAIN": chain,
        "Mut": mut,
    }

    # collect per-suffix node+edge features
    for suf in suffixes:
        nodes_csv = base_dir / f"{pdb_id}_{suf}_nodes.csv"
        edges_csv = base_dir / f"{pdb_id}_{suf}_edges.csv"

        net_kind = "NA" if suf.startswith("NAinter") else "AA"

        # nodes
        node_feats = summarize_nodes(nodes_csv, network_kind=net_kind, suffix=suf)
        features_all.update(node_feats)

        # edges
        edge_feats = summarize_edges(edges_csv, suffix=suf)
        features_all.update(edge_feats)

    # add Δ features for AAinter and NAinter (node + edge)
    add_node_deltas(features_all, "AAinter")
    add_node_deltas(features_all, "NAinter")
    add_edge_deltas(features_all, "AAinter")
    add_edge_deltas(features_all, "NAinter")

    # special Δ_NumberOfUndirectedEdges_NAinter_noweight (amino-only in NA, no weight)
    features_all["Δ_NumberOfUndirectedEdges_NAinter_noweight"] = compute_delta_NumberOfUndirectedEdges_NA_amino_only(base_dir, pdb_id)

    out_csv = base_dir / f"{pdb_id}_network_features.csv"
    pd.DataFrame([features_all]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


# =========================
# Step4: print in order + cleanup
# =========================
def read_features(network_csv: Path) -> dict:
    if not network_csv.exists():
        raise FileNotFoundError(f"Not found: {network_csv}")

    df = pd.read_csv(network_csv)
    if df.empty:
        raise ValueError(f"Empty CSV: {network_csv}")

    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if k in ("PDB", "CHAIN", "Mut"):
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out


def cleanup_outputs(script_dir: Path, pdb_id: str):
    # delete local interaction CSVs
    local_patterns = [
        f"{pdb_id}_AAinter_w*.csv",
        f"{pdb_id}_AAinter_m*.csv",
        f"{pdb_id}_NAinter_w*.csv",
        f"{pdb_id}_NAinter_m*.csv",
    ]
    for pat in local_patterns:
        for fp in script_dir.glob(pat):
            try:
                fp.unlink()
            except Exception:
                pass

    # remove cytoscape_results entirely
    root_dir = script_dir / "cytoscape_results"
    if root_dir.exists():
        try:
            shutil.rmtree(root_dir)
        except Exception:
            pass


# =========================
# Main pipeline
# =========================
def main():
    ap = argparse.ArgumentParser(description="All-in-one network feature extraction (interaction -> Cytoscape -> features -> output).")
    ap.add_argument("-PDB", required=True, dest="pdb_id", help="PDB ID (e.g., 1URN), expects <PDB>.pdb and <PDB>_1.pdb in script dir")
    ap.add_argument("-CHAIN", required=True, dest="chain", help="Protein chain ID (e.g., A)")
    ap.add_argument("-Mut", required=True, dest="mut", help="Mutation (e.g., D92A)")
    args = ap.parse_args()

    pdb_id = args.pdb_id.strip()
    chain = args.chain.strip()
    mut = args.mut.strip()

    script_dir = Path(__file__).resolve().parent

    # Step1
    step1_make_interaction_tables(script_dir, pdb_id, chain, mut)

    # Step2
    step2_cytoscape_analyze(script_dir, pdb_id)

    # Step3
    net_csv = step3_compute_network_features(script_dir, pdb_id, chain, mut)

    # Step4
    feats = read_features(net_csv)
    for name in FEATURE_ORDER:
        print(feats.get(name, 0.0))

    # cleanup
    cleanup_outputs(script_dir, pdb_id)


if __name__ == "__main__":
    main()
