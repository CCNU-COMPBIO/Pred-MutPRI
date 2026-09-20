#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Internal structural-feature module for Pred-MutPRI.

"""

import argparse
import re
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Tuple


MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mut(mut_str: str):
    m = MUT_RE.match(mut_str.strip())
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut_str}. Expected like D92A.")
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def find_foldx_exe(foldx_dir: Path) -> Path:
    candidates = [
        foldx_dir / "FoldX",
        foldx_dir / "foldx",
        foldx_dir / "FoldX.exe",
        foldx_dir / "foldx.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"FoldX executable not found in {foldx_dir}")


def snapshot_files(folder: Path) -> set[Path]:
    files = set()
    for p in folder.rglob("*"):
        if p.is_file():
            files.add(p.relative_to(folder))
    return files


def _debug_enabled() -> bool:
    return os.environ.get("PRED_MUTPRI_DEBUG", "").strip() == "1"


def run_cmd(cmd, cwd: Path) -> str:
    r = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if r.returncode != 0:
        if _debug_enabled():
            raise RuntimeError(
                f"Command failed: {' '.join(map(str, cmd))}\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        raise RuntimeError(
            f"Internal command failed: {Path(str(cmd[0])).name}. "
            "Set PRED_MUTPRI_DEBUG=1 for diagnostics."
        )
    return r.stdout


def run_foldx_buildmodel(
    script_dir: Path,
    pdb_id: str,
    chain: str,
    mut: str,
) -> Path:
    """
    Original FoldX BuildModel workflow:
    - WT PDB is <script_dir>/<PDB>.pdb
    - FoldX is <script_dir>/FoldX/
    - generated non-WT mutant PDB is copied back to script_dir
    - newly generated FoldX working files are cleaned
    """
    foldx_dir = script_dir / "FoldX"
    if not foldx_dir.exists():
        raise FileNotFoundError(f"FoldX directory not found: {foldx_dir}")

    foldx_exe = find_foldx_exe(foldx_dir)

    src_pdb = script_dir / f"{pdb_id}.pdb"
    if not src_pdb.exists():
        raise FileNotFoundError(f"Input PDB not found: {src_pdb}")

    wt, pos, mt = parse_mut(mut)
    before = snapshot_files(foldx_dir)

    local_pdb = foldx_dir / src_pdb.name
    shutil.copy2(src_pdb, local_pdb)

    mut_line = f"{wt}{chain}{pos}{mt};"
    mut_file = foldx_dir / "individual_list.txt"
    mut_file.write_text(mut_line + "\n", encoding="utf-8")

    cmd = [
        str(foldx_exe),
        "--command=BuildModel",
        f"--pdb={local_pdb.name}",
        f"--mutant-file={mut_file.name}",
    ]
    run_cmd(cmd, cwd=foldx_dir)

    after = snapshot_files(foldx_dir)
    new_files = sorted(after - before)

    copied = []
    for rel in new_files:
        p = foldx_dir / rel
        if p.suffix.lower() != ".pdb":
            continue
        if p.name == local_pdb.name:
            continue
        if p.name.upper().startswith("WT"):
            continue
        dst = script_dir / p.name
        shutil.copy2(p, dst)
        copied.append(dst)

    for rel in new_files:
        p = foldx_dir / rel
        try:
            p.unlink()
        except Exception:
            pass

    prefer = script_dir / f"{pdb_id}_1.pdb"
    if prefer.exists():
        return prefer
    if copied:
        return copied[0]

    raise RuntimeError(
        "FoldX finished but no mutant PDB was copied back (non-WT)."
    )


# -------------------------
# DSSP: T-segment ratio
# -------------------------
HEADER_LINES = 28
SS_COL_0BASED = 16
CHAIN_COL_0BASED = 11
AA_COL_0BASED = 13


def run_mkdssp(pdb_path: Path, dssp_path: Path):
    cmd = ["mkdssp", str(pdb_path), str(dssp_path)]
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"mkdssp failed.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )


def compute_T_segment_ratio(dssp_path: Path, chain_id: str) -> float:
    lines = dssp_path.read_text(errors="ignore").splitlines()
    if len(lines) <= HEADER_LINES:
        raise RuntimeError("DSSP file too short.")

    current_seg = None
    prev_was_space = False
    total_segments = 0
    t_segments = 0

    for line in lines[HEADER_LINES:]:
        if len(line) <= max(SS_COL_0BASED, CHAIN_COL_0BASED, AA_COL_0BASED):
            continue

        chain = line[CHAIN_COL_0BASED].strip()
        if chain != chain_id:
            continue

        aa = line[AA_COL_0BASED].strip()
        if aa == "!" or aa == "":
            continue

        ss = line[SS_COL_0BASED]

        if ss == " ":
            if current_seg is not None:
                prev_was_space = True
            continue

        if current_seg is None:
            current_seg = ss
            total_segments += 1
            if ss == "T":
                t_segments += 1
            prev_was_space = False
            continue

        if prev_was_space:
            current_seg = ss
            total_segments += 1
            if ss == "T":
                t_segments += 1
            prev_was_space = False
        elif ss != current_seg:
            current_seg = ss
            total_segments += 1
            if ss == "T":
                t_segments += 1

    if total_segments == 0:
        raise RuntimeError(f"No valid segments for chain {chain_id}")

    return t_segments / total_segments


# -------------------------
# ProtInter ionic Total
# -------------------------
TOTAL_RE = re.compile(
    r"Total:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def _resolve_executable(executable: str, base_dir: Path) -> str:
    """
    Keep plain commands such as ``protinter`` unchanged so PATH lookup works.
    If the user supplies a relative path containing a directory component,
    resolve it against the project directory before changing into a temporary
    working directory.
    """
    exe = Path(executable).expanduser()
    if exe.is_absolute():
        return str(exe)

    if exe.parent != Path("."):
        return str((base_dir / exe).resolve())

    return executable


def write_single_protein_chain_pdb(
    source_pdb: Path,
    chain_id: str,
    output_pdb: Path,
) -> int:
    """
    Write only ATOM records belonging to ``chain_id`` to a temporary PDB.

    ProtInter's ionic-interaction calculation is intended to be run on the
    protein chain itself.  Passing the complete protein-RNA complex can return
    zero interactions for structures that correctly give non-zero values when
    the protein chain is isolated.

    Coordinates, atom names, residue names/numbers, occupancies and B-factors
    are copied byte-for-byte from the source ATOM records.  Only unrelated
    chains and non-ATOM records are omitted.

    Returns the number of ATOM records written.
    """
    atom_count = 0

    with source_pdb.open("r", errors="ignore") as src, output_pdb.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            # PDB fixed columns: chain ID is column 22 (0-based index 21).
            if not line.startswith("ATOM"):
                continue
            if len(line) <= 21:
                continue
            if line[21].strip() != chain_id:
                continue

            dst.write(line.rstrip("\r\n") + "\n")
            atom_count += 1

        dst.write("TER\n")
        dst.write("END\n")

    if atom_count == 0:
        raise ValueError(
            f"No ATOM records found for protein chain {chain_id} "
            f"in {source_pdb}"
        )

    return atom_count


def protinter_ionic_total(
    workdir: Path,
    mutant_pdb: Path,
    chain_id: str,
    protinter_exe: str = "protinter",
) -> float:
    """
    Compute the mutant protein-chain ionic interaction total.

    The complete complex is *not* passed to ProtInter.  A temporary PDB
    containing only the mutated protein chain is created in the system
    temporary directory, used once, and automatically removed.
    """
    tmp_base = os.environ.get("PRED_MUTPRI_TMPDIR") or None
    resolved_exe = _resolve_executable(protinter_exe, workdir)

    with tempfile.TemporaryDirectory(
        prefix="predmutpri_protinter_",
        dir=tmp_base,
    ) as tmp_name:
        tmp_dir = Path(tmp_name)
        try:
            tmp_dir.chmod(0o700)
        except Exception:
            pass

        chain_pdb = tmp_dir / f"{mutant_pdb.stem}_chain_{chain_id}.pdb"
        write_single_protein_chain_pdb(
            mutant_pdb,
            chain_id,
            chain_pdb,
        )

        cmd = [resolved_exe, "--ionic", chain_pdb.name]
        out = run_cmd(cmd, cwd=tmp_dir)

        m = TOTAL_RE.search(out)
        if not m:
            if _debug_enabled():
                raise ValueError(
                    "Could not find 'Total:' in ProtInter output.\n"
                    f"ProtInter output:\n{out}"
                )
            raise ValueError(
                "Could not find 'Total:' in ProtInter output. "
                "Set PRED_MUTPRI_DEBUG=1 for diagnostics."
            )

        return float(m.group(1))


# -------------------------
# PDB resolution
# -------------------------
RES_RE = re.compile(
    r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS\.",
    re.IGNORECASE,
)


def parse_resolution(pdb_path: Path):
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            m = RES_RE.match(line.strip())
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


# -------------------------
# DSSR/SNAP features
# -------------------------
RE_PHOS_HB = re.compile(
    r"^List\s+of\s+(\d+)\s+phosphate/amino-acid H-bonds",
    re.IGNORECASE,
)
RE_NUC_AA = re.compile(
    r"^List\s+of\s+(\d+)\s+nucleotide/amino-acid interactions",
    re.IGNORECASE,
)
RE_BP_AA = re.compile(
    r"^List\s+of\s+(\d+)\s+base-pair/amino-acid interactions",
    re.IGNORECASE,
)


def extract_value_from_txt(txt_path: Path, pattern: re.Pattern) -> int:
    if not txt_path.exists():
        return 0
    with txt_path.open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("List"):
                continue
            m = pattern.match(s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return 0
    return 0


def dssr_features(
    script_dir: Path,
    wt_pdb: Path,
    mut_pdb: Path,
):
    dssr_exe = script_dir / "x3dna-dssr" / "x3dna-dssr"
    if not dssr_exe.exists():
        dssr_exe_win = script_dir / "x3dna-dssr" / "x3dna-dssr.exe"
        if dssr_exe_win.exists():
            dssr_exe = dssr_exe_win
        else:
            raise FileNotFoundError(
                f"x3dna-dssr executable not found in "
                f"{script_dir / 'x3dna-dssr'}"
            )

    tmp_base = os.environ.get("PRED_MUTPRI_TMPDIR") or None

    with tempfile.TemporaryDirectory(
        prefix="predmutpri_dssr_",
        dir=tmp_base,
    ) as tmp_name:
        tmp_dir = Path(tmp_name)
        try:
            tmp_dir.chmod(0o700)
        except Exception:
            pass

        wt_txt = tmp_dir / "wt.txt"
        mut_txt = tmp_dir / "mut.txt"

        run_cmd(
            [str(dssr_exe), "snap", f"-i={wt_pdb}", f"-o={wt_txt}"],
            cwd=tmp_dir,
        )
        run_cmd(
            [str(dssr_exe), "snap", f"-i={mut_pdb}", f"-o={mut_txt}"],
            cwd=tmp_dir,
        )

        phos_hb_wt = extract_value_from_txt(wt_txt, RE_PHOS_HB)

        nuc_wt = extract_value_from_txt(wt_txt, RE_NUC_AA)
        nuc_mut = extract_value_from_txt(mut_txt, RE_NUC_AA)
        delta_nuc = nuc_mut - nuc_wt

        bp_wt = extract_value_from_txt(wt_txt, RE_BP_AA)
        bp_mut = extract_value_from_txt(mut_txt, RE_BP_AA)
        delta_bp = bp_mut - bp_wt

        return phos_hb_wt, delta_nuc, delta_bp


def compute_structural_features(
    pdb_id: str,
    chain: str,
    mut: str,
    protinter_exe: str = "protinter",
    script_dir: Path | None = None,
) -> Tuple[List[float], Path]:
    """
    Return:
        ([6 structural features in the original order], mutant_pdb_path)

    No feature value is printed or written to a feature file.
    """
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path(script_dir).resolve()

    pdb_id = pdb_id.strip()
    chain = chain.strip()
    mut = mut.strip()

    if len(chain) != 1:
        raise ValueError("CHAIN must be 1 character")

    wt_pdb = script_dir / f"{pdb_id}.pdb"
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB not found: {wt_pdb}")

    mut_pdb = run_foldx_buildmodel(script_dir, pdb_id, chain, mut)

    dssp_path = script_dir / f"{pdb_id}.dssp"
    try:
        run_mkdssp(wt_pdb, dssp_path)
        sse_t_ratio = compute_T_segment_ratio(dssp_path, chain)
    finally:
        if dssp_path.exists():
            try:
                dssp_path.unlink()
            except Exception:
                pass

    ionic_total = protinter_ionic_total(
        script_dir,
        mut_pdb,
        chain,
        protinter_exe=protinter_exe,
    )

    resolution = parse_resolution(wt_pdb)
    resolution = 0.0 if resolution is None else float(resolution)

    phos_hb_wt, delta_nuc, delta_bp = dssr_features(
        script_dir,
        wt_pdb,
        mut_pdb,
    )

    features = [
        float(sse_t_ratio),
        float(ionic_total),
        float(resolution),
        float(phos_hb_wt),
        float(delta_nuc),
        float(delta_bp),
    ]
    return features, mut_pdb


def main():
    parser = argparse.ArgumentParser(
        description="Internal Pred-MutPRI structural feature component."
    )
    parser.add_argument("-PDB", required=True)
    parser.add_argument("-CHAIN", required=True)
    parser.add_argument("-Mut", required=True)
    parser.parse_args()

    raise SystemExit(
        "structural_feature.py is an internal component. "
        "Use ./predict.sh for prediction."
    )


if __name__ == "__main__":
    main()
