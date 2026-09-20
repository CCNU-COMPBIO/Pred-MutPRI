#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Internal, memory-only network feature module for Pred-MutPRI.

"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import py4cytoscape as p4c
from Bio.PDB import PDBParser


FEATURE_ORDER = [
    "AverageShortestPathLength_Weightedinter_w",
    "Δ_NumberOfUndirectedEdges_AA_inter",
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

NODE_FEATURES = [
    "AverageShortestPathLength",
    "ClusteringCoefficient",
    "ClosenessCentrality",
    "PartnerOfMultiEdgedNodePairs",
    "SelfLoops",
    "Eccentricity",
    "Stress",
    "Degree",
    "BetweennessCentrality",
    "NeighborhoodConnectivity",
    "NumberOfDirectedEdges",
    "NumberOfUndirectedEdges",
    "Radiality",
    "TopologicalCoefficient",
]

PARSER = PDBParser(QUIET=True)

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
NT_NAMES = {"A", "U", "G", "C", "DA", "DT", "DG", "DC"}


def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s).strip().strip(".")
    return s[:180] if len(s) > 180 else s


def parse_mut(mut: str) -> Tuple[str, int, str]:
    m = re.fullmatch(r"([A-Za-z])(\d+)([A-Za-z])", mut.strip())
    if not m:
        raise ValueError(
            f"Invalid -Mut format: {mut} (expected like D92A)"
        )
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def get_structure(pdb_path: Path):
    return PARSER.get_structure(pdb_path.stem, str(pdb_path))


def is_standard_residue(res) -> bool:
    return (
        res.get_resname().strip().upper() in AA3
        and res.id[0] == " "
    )


def is_nucleotide(res) -> bool:
    return res.get_resname().strip().upper() in NT_NAMES


def residue_uid(chain_id: str, res) -> str:
    resname = res.get_resname().strip().upper()
    resseq = int(res.id[1])
    return f"{resname}_{chain_id}{resseq}"


def atoms_of_residue(res):
    # Preserve all atoms present in the PDB, including hydrogens.
    return list(res.get_atoms())


def find_residue(structure, chain_id: str, pos: int):
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(
            f"Chain {chain_id} not found in structure {structure.id}"
        )
    chain = model[chain_id]
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if int(res.id[1]) == pos:
            return chain, res
    raise ValueError(
        f"Residue position {pos} not found in chain {chain_id}"
    )


def add_source_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()

    if kind == "AA":
        c1, c2, c3, c4 = (
            "Target_Residue",
            "Target_Atom",
            "Neighbor_Residue",
            "Neighbor_Atom",
        )
    elif kind == "NA":
        c1, c2, c3, c4 = (
            "Protein_Residue",
            "Protein_Atom",
            "Nucleic_Residue",
            "Nucleic_Atom",
        )
    else:
        raise ValueError("kind must be 'AA' or 'NA'")

    if df.empty:
        df["source"] = pd.Series(dtype=str)
        df["target"] = pd.Series(dtype=str)
    else:
        df["source"] = (
            df[c1].astype(str) + "_" + df[c2].astype(str)
        )
        df["target"] = (
            df[c3].astype(str) + "_" + df[c4].astype(str)
        )

    base_cols = [
        c for c in df.columns
        if c not in ("source", "target")
    ]
    return df[base_cols + ["source", "target"]]


def extract_aa_aa_contacts(
    pdb_path: Path,
    chain_id: str,
    mut_pos: int,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    columns = [
        "Target_Residue",
        "Target_Atom",
        "Neighbor_Residue",
        "Neighbor_Atom",
        "Distance",
    ]
    if not mut_atoms:
        return add_source_target(
            pd.DataFrame(columns=columns),
            kind="AA",
        )

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

    df = pd.DataFrame(rows, columns=columns)
    return add_source_target(df, kind="AA")


def extract_aa_na_contacts(
    pdb_path: Path,
    chain_id: str,
    mut_pos: int,
    cutoff: float = 10.0,
) -> pd.DataFrame:
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    columns = [
        "Protein_Residue",
        "Protein_Atom",
        "Nucleic_Residue",
        "Nucleic_Atom",
        "Distance",
    ]
    if not mut_atoms:
        return add_source_target(
            pd.DataFrame(columns=columns),
            kind="NA",
        )

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

            nt_uid = residue_uid(ch_id, res)
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

    df = pd.DataFrame(rows, columns=columns)
    return add_source_target(df, kind="NA")


def build_interaction_frames(
    wt_pdb: Path,
    mut_pdb: Path,
    chain: str,
    mut: str,
) -> Dict[str, pd.DataFrame]:
    _, mut_pos, _ = parse_mut(mut)

    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB not found: {wt_pdb}")
    if not mut_pdb.exists():
        raise FileNotFoundError(f"Mutant PDB not found: {mut_pdb}")

    return {
        "AAinter_w": extract_aa_aa_contacts(
            wt_pdb, chain, mut_pos, cutoff=5.0
        ),
        "AAinter_m": extract_aa_aa_contacts(
            mut_pdb, chain, mut_pos, cutoff=5.0
        ),
        "NAinter_w": extract_aa_na_contacts(
            wt_pdb, chain, mut_pos, cutoff=10.0
        ),
        "NAinter_m": extract_aa_na_contacts(
            mut_pdb, chain, mut_pos, cutoff=10.0
        ),
    }


def edge_frame_from_interactions(
    interaction_df: pd.DataFrame,
) -> pd.DataFrame:
    if interaction_df is None or interaction_df.empty:
        return pd.DataFrame(columns=["source", "target"])

    if "source" in interaction_df.columns and "target" in interaction_df.columns:
        edges = interaction_df[["source", "target"]].copy()
    else:
        if interaction_df.shape[1] < 2:
            return pd.DataFrame(columns=["source", "target"])
        edges = pd.DataFrame({
            "source": interaction_df.iloc[:, -2].astype(str),
            "target": interaction_df.iloc[:, -1].astype(str),
        })

    edges = edges.dropna()
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges = edges[
        (edges["source"] != "")
        & (edges["target"] != "")
    ]
    return edges.reset_index(drop=True)


def create_and_analyze_network(
    edges: pd.DataFrame,
    net_name: str,
    directed: bool = False,
) -> int:
    """
    Create a Cytoscape network WITHOUT py4cytoscape's
    create_network_from_data_frames() helper.

    Reason:
    create_network_from_data_frames() automatically applies the default
    visual style (vizmap) and preferred layout. Pred-MutPRI only needs
    Analyzer topology tables, so those visualization steps are unnecessary
    and can trigger "_delay_until_stable(): Timeout trying to apply vizmap".

    Here we create the network directly from CytoscapeJS JSON via CyREST,
    wait briefly for the network model/table to become available, set it
    current, and run Analyzer. No style or layout is applied.
    """
    if edges is None or edges.empty:
        raise ValueError("Cannot analyze an empty edge table.")

    edge_df = edges[["source", "target"]].copy()
    edge_df["source"] = edge_df["source"].astype(str)
    edge_df["target"] = edge_df["target"].astype(str)

    node_ids = pd.unique(
        pd.concat(
            [edge_df["source"], edge_df["target"]],
            ignore_index=True,
        )
    ).tolist()

    json_nodes = [
        {
            "data": {
                "id": str(node_id),
                "name": str(node_id),
            }
        }
        for node_id in node_ids
    ]

    json_edges = []
    for i, row in edge_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        interaction = "interacts with"
        json_edges.append(
            {
                "data": {
                    "id": f"e{i}",
                    "name": f"{source} ({interaction}) {target}",
                    "source": source,
                    "target": target,
                    "interaction": interaction,
                }
            }
        )

    cytoscapejs = {
        "data": {"name": net_name},
        "elements": {
            "nodes": json_nodes,
            "edges": json_edges,
        },
    }

    # This lower-level helper only POSTs the network JSON.
    # It does NOT run "vizmap apply" or a layout.
    net_suid = p4c.networks.create_network_from_cytoscapejs(
        cytoscapejs,
        title=net_name,
        collection="Pred-MutPRI_internal",
    )

    # Cytoscape updates its model asynchronously. Wait only for the network
    # object/table to become accessible; do not wait for any visual style.
    last_error = None
    for _ in range(50):
        try:
            p4c.networks.get_network_suid(net_suid)
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        raise RuntimeError(
            f"Cytoscape network model did not become ready: {last_error}"
        )

    try:
        p4c.networks.set_current_network(net_suid)
    except Exception:
        pass

    try:
        p4c.analyze_network(
            network=net_suid,
            directed=directed,
        )
    except TypeError:
        p4c.networks.set_current_network(net_suid)
        p4c.analyze_network()

    return net_suid


def analyze_interaction_frame(
    interaction_df: pd.DataFrame,
    net_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze one network in Cytoscape and retrieve node/edge tables in memory.

    No CSV is written.
    """
    edges = edge_frame_from_interactions(interaction_df)
    if edges.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    net_suid = None
    try:
        net_suid = create_and_analyze_network(
            edges,
            net_name=net_name,
            directed=False,
        )

        try:
            node_table = p4c.tables.get_table_columns(
                "node",
                network=net_suid,
            )
            edge_table = p4c.tables.get_table_columns(
                "edge",
                network=net_suid,
            )
        except TypeError:
            node_table = p4c.tables.get_table_columns("node")
            edge_table = p4c.tables.get_table_columns("edge")

        return (
            pd.DataFrame(node_table),
            pd.DataFrame(edge_table),
        )
    except Exception:
        # Preserve the original pipeline behavior: if Cytoscape cannot
        # analyze one of the four networks (for example, a network with
        # fewer than four nodes), the downstream features for that network
        # are computed from empty tables and therefore default to 0.
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )
    finally:
        if net_suid is not None:
            try:
                p4c.networks.delete_network(net_suid)
            except Exception:
                pass


def classify_node_type(name: str) -> str:
    try:
        prefix = str(name).split("_")[0].strip()
        if len(prefix) <= 2:
            return "nuc"
        if len(prefix) == 3:
            return "amino"
        return "unknown"
    except Exception:
        return "unknown"


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


def summarize_nodes_df(
    df: pd.DataFrame,
    network_kind: str,
    suffix: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for feat in NODE_FEATURES:
        out[f"{feat}_{suffix}"] = 0.0

    out[f"n_amino_{suffix}"] = 0.0
    out[f"n_nuc_{suffix}"] = 0.0
    out[f"n_unknown_{suffix}"] = 0.0

    if df is None or df.empty or "name" not in df.columns:
        return out

    df = df.copy()
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

    if network_kind.upper() == "NA":
        w_aa, w_na = 0.6, 0.4
    else:
        w_aa, w_na = 1.0, 0.0

    for feat in NODE_FEATURES:
        aa_mean = (
            safe_mean(aa_df[feat]) if len(aa_df) else 0.0
        )
        na_mean = (
            safe_mean(na_df[feat]) if len(na_df) else 0.0
        )
        out[f"{feat}_{suffix}"] = float(
            w_aa * aa_mean + w_na * na_mean
        )

    return out


def extract_element(atom_name: str) -> str:
    atom_name = str(atom_name).strip()
    if "_" in atom_name:
        last = atom_name.split("_")[-1]
        if last:
            return last[0].upper()
    return "Unrecognized"


def classify_interaction(source: str, target: str) -> str:
    src = extract_element(source)
    tgt = extract_element(target)

    if "Unrecognized" in (src, tgt):
        return "Unrecognized"

    pair = {src, tgt}

    if pair in [{"H", "O"}, {"H", "N"}]:
        return "H–O/H–N"

    if pair in [{"N", "O"}, {"C", "O"}, {"C", "N"}]:
        return "N–O/C–O/C–N"

    return "Other"


def summarize_edges_df(
    df: pd.DataFrame,
    suffix: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    base_names = [
        "Total_interactions",
        "H-O/H-N_Count",
        "N-O/C-O/C-N_Count",
        "Other_Count",
        "Unrec_Count",
        "H-O/H-N_Mean",
        "N-O/C-O/C-N_Mean",
        "Other_Mean",
        "H-O/H-N_Ratio",
        "N-O/C-O/C-N_Ratio",
        "Other_Ratio",
        "Evaluation",
    ]

    for name in base_names:
        out[f"{name}_{suffix}"] = 0.0

    if df is None or df.empty:
        return out

    required = {"source", "target", "EdgeBetweenness"}
    if not required.issubset(df.columns):
        return out

    df = df.copy()
    df["EdgeBetweenness"] = pd.to_numeric(
        df["EdgeBetweenness"],
        errors="coerce",
    )
    df["__type__"] = df.apply(
        lambda row: classify_interaction(
            row["source"],
            row["target"],
        ),
        axis=1,
    )

    total = len(df)
    counts = df["__type__"].value_counts().to_dict()

    def mean_for(t: str) -> float:
        x = df.loc[
            df["__type__"] == t,
            "EdgeBetweenness",
        ].dropna()
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

    denom = float(total) if total else 1.0
    out[f"H-O/H-N_Ratio_{suffix}"] = float(hon / denom)
    out[f"N-O/C-O/C-N_Ratio_{suffix}"] = float(nocon / denom)
    out[f"Other_Ratio_{suffix}"] = float(oth / denom)

    out[f"Evaluation_{suffix}"] = float(
        0.6 * mean_h
        + 0.3 * mean_no
        + 0.1 * mean_other
    )

    return out


def add_edge_deltas(
    features: Dict[str, float],
    net: str,
) -> None:
    edge_bases = [
        "Total_interactions",
        "H-O/H-N_Count",
        "N-O/C-O/C-N_Count",
        "Other_Count",
        "Unrec_Count",
        "H-O/H-N_Mean",
        "N-O/C-O/C-N_Mean",
        "Other_Mean",
        "H-O/H-N_Ratio",
        "N-O/C-O/C-N_Ratio",
        "Other_Ratio",
        "Evaluation",
    ]

    for name in edge_bases:
        v_m = float(features.get(f"{name}_{net}_m", 0.0))
        v_w = float(features.get(f"{name}_{net}_w", 0.0))
        features[f"Δ_{name}_{net}"] = v_m - v_w


def add_node_deltas(
    features: Dict[str, float],
    net: str,
) -> None:
    for feat in NODE_FEATURES:
        v_m = float(features.get(f"{feat}_{net}_m", 0.0))
        v_w = float(features.get(f"{feat}_{net}_w", 0.0))
        features[f"Δ_{feat}_{net}"] = v_m - v_w


def amino_only_undirected_mean(
    nodes_df: pd.DataFrame,
) -> float:
    if (
        nodes_df is None
        or nodes_df.empty
        or "name" not in nodes_df.columns
        or "NumberOfUndirectedEdges" not in nodes_df.columns
    ):
        return 0.0

    df = nodes_df.copy()
    df["__type__"] = df["name"].apply(classify_node_type)
    aa = df[df["__type__"] == "amino"]

    if aa.empty:
        return 0.0

    return safe_mean(aa["NumberOfUndirectedEdges"])


def compute_network_features(
    pdb_id: str,
    chain: str,
    mut: str,
    script_dir: Path | None = None,
    wt_pdb: Path | None = None,
    mut_pdb: Path | None = None,
) -> List[float]:
    """
    Return the formal 30 network features in model order.

    All intermediate interaction/node/edge tables remain in memory.
    """
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path(script_dir).resolve()

    pdb_id = pdb_id.strip()
    chain = chain.strip()
    mut = mut.strip()

    if wt_pdb is None:
        wt_pdb = script_dir / f"{pdb_id}.pdb"
    else:
        wt_pdb = Path(wt_pdb)

    if mut_pdb is None:
        mut_pdb = script_dir / f"{pdb_id}_1.pdb"
    else:
        mut_pdb = Path(mut_pdb)

    frames = build_interaction_frames(
        wt_pdb,
        mut_pdb,
        chain,
        mut,
    )

    p4c.cytoscape_ping()

    features_all: Dict[str, float] = {}
    na_w_amino_undirected = 0.0
    na_m_amino_undirected = 0.0

    for suffix in (
        "AAinter_w",
        "AAinter_m",
        "NAinter_w",
        "NAinter_m",
    ):
        node_df, edge_df = analyze_interaction_frame(
            frames[suffix],
            net_name=safe_filename(f"{pdb_id}_{suffix}"),
        )

        net_kind = (
            "NA" if suffix.startswith("NAinter") else "AA"
        )

        features_all.update(
            summarize_nodes_df(
                node_df,
                network_kind=net_kind,
                suffix=suffix,
            )
        )
        features_all.update(
            summarize_edges_df(
                edge_df,
                suffix=suffix,
            )
        )

        if suffix == "NAinter_w":
            na_w_amino_undirected = (
                amino_only_undirected_mean(node_df)
            )
        elif suffix == "NAinter_m":
            na_m_amino_undirected = (
                amino_only_undirected_mean(node_df)
            )

        # Explicitly release large tables as soon as possible.
        del node_df, edge_df

    add_node_deltas(features_all, "AAinter")
    add_node_deltas(features_all, "NAinter")
    add_edge_deltas(features_all, "AAinter")
    add_edge_deltas(features_all, "NAinter")

    features_all[
        "AverageShortestPathLength_Weightedinter_w"
    ] = float(
        features_all.get(
            "AverageShortestPathLength_NAinter_w",
            0.0,
        )
    )

    features_all[
        "Δ_NumberOfUndirectedEdges_AA_inter"
    ] = float(
        na_m_amino_undirected - na_w_amino_undirected
    )

    result = [
        float(features_all.get(name, 0.0))
        for name in FEATURE_ORDER
    ]

    if len(result) != 30:
        raise RuntimeError(
            f"Internal network feature count error: {len(result)}"
        )

    return result


def main():
    ap = argparse.ArgumentParser(
        description="Internal Pred-MutPRI network component."
    )
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    ap.parse_args()

    raise SystemExit(
        "run_network.py is an internal component. "
        "Use ./predict.sh for prediction."
    )


if __name__ == "__main__":
    main()
