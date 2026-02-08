#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_and_parse_floats(cmd, cwd: Path) -> list[float]:
    r = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd))}\n"
            f"cwd={cwd}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

    out = (r.stdout or "").strip()
    if not out:
        return []

    vals: list[float] = []
    for tok in out.replace("\t", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def load_xgb_booster_json(model_json: Path):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(model_json))
    return booster


def main():
    ap = argparse.ArgumentParser(description="38 features (struct 6 + network 30 + ES + ESM) -> XGBoost prediction.")
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)

    ap.add_argument("--model_json", required=True, help="Path to XGBoost Booster JSON model (e.g. model.json)")

    # specify interpreters (multi-env)
    ap.add_argument("--py_struct", default=sys.executable, help="Python for structural_feature.py")
    ap.add_argument("--py_network", default=sys.executable, help="Python for run_network.py")
    ap.add_argument("--py_esm", default=sys.executable, help="Python for run_esm.py")
    ap.add_argument("--bash", default="bash", help="Bash executable for run_af3_all.sh")

    ap.add_argument("--protinter", default="protinter", help="protinter executable")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    pdb = args.PDB.strip()
    chain = args.CHAIN.strip()
    mut = args.Mut.strip()

    model_json = Path(args.model_json).expanduser().resolve()
    booster = load_xgb_booster_json(model_json)

    # 1) structural_feature: 6 floats
    structural_py = script_dir / "structural_feature.py"
    if not structural_py.exists():
        raise FileNotFoundError(f"Missing {structural_py}. Rename structural_feature(1).py -> structural_feature.py")

    f1 = run_and_parse_floats(
        [args.py_struct, str(structural_py), "-PDB", pdb, "-CHAIN", chain, "-Mut", mut, "--protinter", args.protinter],
        cwd=script_dir,
    )
    if len(f1) != 6:
        raise ValueError(f"structural_feature.py must output 6 floats, got {len(f1)}: {f1}")

    # 2) network: 30 floats
    network_py = script_dir / "run_network.py"
    if not network_py.exists():
        raise FileNotFoundError(f"Missing {network_py}. Rename run_network(1).py -> run_network.py")

    f2 = run_and_parse_floats(
        [args.py_network, str(network_py), "-PDB", pdb, "-CHAIN", chain, "-Mut", mut],
        cwd=script_dir,
    )
    if len(f2) != 30:
        raise ValueError(f"run_network.py must output 30 floats, got {len(f2)}")

    # 3) ES (AF3): 1 float
    af3_sh = script_dir / "run_af3_all.sh"
    if not af3_sh.exists():
        raise FileNotFoundError(f"Missing {af3_sh}")

    f3 = run_and_parse_floats(
        [args.bash, str(af3_sh), "-PDB", pdb, "-CHAIN", chain, "-Mut", mut],
        cwd=script_dir,
    )
    if not f3:
        raise ValueError("run_af3_all.sh produced no numeric output (expected Avg_ES)")
    es = f3[-1]

    # 4) ESM: 1 float
    esm_py = script_dir / "run_esm.py"
    if not esm_py.exists():
        raise FileNotFoundError(f"Missing {esm_py}. Rename run_esm(2).py -> run_esm.py")

    f4 = run_and_parse_floats(
        [args.py_esm, str(esm_py), "-PDB", pdb, "-CHAIN", chain, "-Mut", mut],
        cwd=script_dir,
    )
    if not f4:
        raise ValueError("run_esm.py produced no numeric output (expected 1 float)")
    esm_val = f4[-1]

    feats = np.array(f1 + f2 + [es] + [esm_val], dtype=float)
    if feats.size != 38:
        raise ValueError(f"Expected 38 features, got {feats.size}")

    # predict with Booster JSON
    import xgboost as xgb
    dmat = xgb.DMatrix(feats.reshape(1, -1))
    pred = booster.predict(dmat)
    y0 = float(np.asarray(pred).ravel()[0])

    # final: only print prediction
    print(f"{y0:.6f}")


if __name__ == "__main__":
    main()