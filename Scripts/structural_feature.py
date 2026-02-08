# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:57:43 2026

@author: xuwan
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# =========================
# FoldX helpers
# =========================
MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

def parse_mut(mut_str: str):
    m = MUT_RE.match(mut_str.strip())
    if not m:
        raise ValueError(f"Invalid -Mut format: {mut_str}. Expected like D92A.")
    wt, pos, mt = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    return wt, pos, mt

def find_foldx_exe(foldx_dir: Path) -> Path:
    candidates = [foldx_dir/"FoldX", foldx_dir/"foldx", foldx_dir/"FoldX.exe", foldx_dir/"foldx.exe"]
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

def run_cmd(cmd, cwd: Path):
    r = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd))}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return r.stdout

def run_foldx_buildmodel(script_dir: Path, pdb_id: str, chain: str, mut: str) -> Path:
    """
    在 script_dir/FoldX 中运行 FoldX BuildModel。
    - 输入 PDB: script_dir/<pdb_id>.pdb
    - 输出：复制突变体PDB到 script_dir（跳过 WT*），清理 FoldX 目录新增文件
    返回：突变体PDB路径（优先 <pdb_id>_1.pdb）
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

    # copy input pdb into FoldX dir (same name)
    local_pdb = foldx_dir / src_pdb.name
    shutil.copy2(src_pdb, local_pdb)

    # write individual_list
    mut_line = f"{wt}{chain}{pos}{mt};"
    mut_file = foldx_dir / "individual_list.txt"
    mut_file.write_text(mut_line + "\n", encoding="utf-8")

    # run FoldX
    cmd = [str(foldx_exe), "--command=BuildModel", f"--pdb={local_pdb.name}", f"--mutant-file={mut_file.name}"]
    run_cmd(cmd, cwd=foldx_dir)

    after = snapshot_files(foldx_dir)
    new_files = sorted(list(after - before))

    # copy mutant pdbs back (skip WT* and the copied input pdb)
    copied = []
    for rel in new_files:
        p = foldx_dir / rel
        if p.suffix.lower() == ".pdb":
            if p.name == local_pdb.name:
                continue
            if p.name.upper().startswith("WT"):
                continue
            dst = script_dir / p.name
            shutil.copy2(p, dst)
            copied.append(dst)

    # clean generated files in FoldX dir (preserve installation)
    for rel in new_files:
        p = foldx_dir / rel
        try:
            p.unlink()
        except Exception:
            pass

    # pick mutant pdb
    prefer = script_dir / f"{pdb_id}_1.pdb"
    if prefer.exists():
        return prefer
    if copied:
        return copied[0]
    raise RuntimeError("FoldX finished but no mutant pdb was copied back (non-WT).")


# =========================
# DSSP: T segment ratio (your segment rule)
# =========================
HEADER_LINES = 28
SS_COL_0BASED = 16
CHAIN_COL_0BASED = 11
AA_COL_0BASED = 13

def run_mkdssp(pdb_path: Path, dssp_path: Path):
    cmd = ["mkdssp", str(pdb_path), str(dssp_path)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mkdssp failed.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

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
        else:
            if ss != current_seg:
                current_seg = ss
                total_segments += 1
                if ss == "T":
                    t_segments += 1

    if total_segments == 0:
        raise RuntimeError(f"No valid segments for chain {chain_id}")
    return t_segments / total_segments


# =========================
# ProtInter: ionic Total
# =========================
TOTAL_RE = re.compile(r"Total:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def protinter_ionic_total(workdir: Path, mutant_pdb: Path, protinter_exe: str = "protinter") -> float:
    cmd = [protinter_exe, "--ionic", mutant_pdb.name]
    out = run_cmd(cmd, cwd=workdir)
    m = TOTAL_RE.search(out)
    if not m:
        raise ValueError("Could not find 'Total:' in protinter output.")
    return float(m.group(1))


# =========================
# PDB resolution (REMARK 2)
# =========================
RES_RE = re.compile(r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS\.", re.IGNORECASE)

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


# =========================
# DSSR snap features
# =========================
RE_PHOS_HB = re.compile(r"^List\s+of\s+(\d+)\s+phosphate/amino-acid H-bonds", re.IGNORECASE)
RE_NUC_AA  = re.compile(r"^List\s+of\s+(\d+)\s+nucleotide/amino-acid interactions", re.IGNORECASE)
RE_BP_AA   = re.compile(r"^List\s+of\s+(\d+)\s+base-pair/amino-acid interactions", re.IGNORECASE)

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

def dssr_features(script_dir: Path, wt_pdb: Path, mut_pdb: Path):
    dssr_exe = script_dir / "x3dna-dssr" / "x3dna-dssr"
    if not dssr_exe.exists():
        dssr_exe_win = script_dir / "x3dna-dssr" / "x3dna-dssr.exe"
        if dssr_exe_win.exists():
            dssr_exe = dssr_exe_win
        else:
            raise FileNotFoundError(f"x3dna-dssr executable not found in {script_dir/'x3dna-dssr'}")

    tmp_dir = script_dir / f"_tmp_dssr_{wt_pdb.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wt_txt = tmp_dir / "wt.txt"
    mut_txt = tmp_dir / "mut.txt"
    try:
        run_cmd([str(dssr_exe), "snap", f"-i={wt_pdb}",  f"-o={wt_txt}"], cwd=tmp_dir)
        run_cmd([str(dssr_exe), "snap", f"-i={mut_pdb}", f"-o={mut_txt}"], cwd=tmp_dir)

        phos_hb_wt = extract_value_from_txt(wt_txt, RE_PHOS_HB)

        nuc_wt = extract_value_from_txt(wt_txt, RE_NUC_AA)
        nuc_mut = extract_value_from_txt(mut_txt, RE_NUC_AA)
        delta_nuc = nuc_mut - nuc_wt

        bp_wt = extract_value_from_txt(wt_txt, RE_BP_AA)
        bp_mut = extract_value_from_txt(mut_txt, RE_BP_AA)
        delta_bp = bp_mut - bp_wt

        return phos_hb_wt, delta_nuc, delta_bp
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="One-pass feature extraction (FoldX + DSSP + ProtInter + Resolution + DSSR)")
    parser.add_argument("-PDB", required=True, help="PDB ID, expects <PDB>.pdb in script directory")
    parser.add_argument("-CHAIN", required=True, help="Protein chain ID (single char)")
    parser.add_argument("-Mut", required=True, help="Mutation WTPosMut (e.g., D92A)")
    parser.add_argument("--protinter", default="protinter", help="protinter executable (default: protinter)")
    parser.add_argument("--keep_mutant_pdb", action="store_true", help="Keep <PDB>_1.pdb after printing features")
    args = parser.parse_args()

    pdb_id = args.PDB.strip()
    chain = args.CHAIN.strip()
    if len(chain) != 1:
        raise ValueError("CHAIN must be 1 character")

    script_dir = Path(__file__).resolve().parent
    wt_pdb = script_dir / f"{pdb_id}.pdb"
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB not found: {wt_pdb}")

    # (1) FoldX -> mutant pdb
    mut_pdb = run_foldx_buildmodel(script_dir, pdb_id, chain, args.Mut)

    # (2) DSSP SSE_T ratio
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

    # (3) ProtInter ionic total (use mutant pdb)
    ionic_total = protinter_ionic_total(script_dir, mut_pdb, protinter_exe=args.protinter)

    # (4) Resolution
    res = parse_resolution(wt_pdb)
    res_out = "0.0" if res is None else str(res)

    # (5) DSSR features
    phos_hb_wt, delta_nuc, delta_bp = dssr_features(script_dir, wt_pdb, mut_pdb)

    # Output in order (one line)
    print(f"{sse_t_ratio} {ionic_total} {res_out} {phos_hb_wt} {delta_nuc} {delta_bp}")

    # Optional cleanup of mutant pdb
    if args.keep_mutant_pdb:
        # only delete the mutant pdb we used (do not touch WT)
        try:
            if mut_pdb.exists():
                mut_pdb.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
