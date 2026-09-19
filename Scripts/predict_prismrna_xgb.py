#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pred-MutPRI top-level predictor.

Feature exposure is minimized as follows:
- structural, network, and ESM features are computed by imported Python
  functions and remain in the same process memory;
- network intermediates remain as DataFrames only;
- AF3-derived ES is returned through an anonymous OS pipe from the AF3 conda
  environment;
- no feature JSON/CSV/text file is created;
- only the final ΔΔG prediction is printed to stdout.

Security boundary:
Pure Python source distributed to a local user cannot cryptographically hide
the feature calculations. A user who can edit/debug/import this source can
instrument it. This design prevents accidental/normal feature exposure, not a
determined local reverse engineer.
"""

import argparse
import contextlib
import io
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Any

import numpy as np

from structural_feature import compute_structural_features
from run_network import compute_network_features
from run_esm import compute_esm_feature


def load_xgb_booster_json(model_json: Path):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_json))
    return booster


def quiet_call(
    func: Callable[..., Any],
    *args,
    **kwargs,
):
    """
    Suppress normal chatter from third-party libraries during successful
    internal computation. On error, include captured diagnostic text.
    """
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            return func(*args, **kwargs)
    except Exception as exc:
        captured = (
            out_buf.getvalue()
            + err_buf.getvalue()
        ).strip()

        if os.environ.get("PRED_MUTPRI_DEBUG", "").strip() == "1" and captured:
            raise RuntimeError(
                f"{exc}\nInternal diagnostics:\n{captured}"
            ) from exc

        raise RuntimeError(
            f"{exc}\n"
            "Set PRED_MUTPRI_DEBUG=1 for internal diagnostics."
        ) from exc


def run_af3_es_via_pipe(
    bash_exe: str,
    af3_script: Path,
    pdb: str,
    chain: str,
    mut: str,
    cwd: Path,
) -> float:
    """
    Run AF3 in its own conda environment and receive exactly one scalar through
    an inherited anonymous pipe. No feature temp file and no stdout parsing.
    """
    if os.name != "posix":
        raise RuntimeError(
            "The anonymous-FD AF3 channel requires Linux/POSIX."
        )

    read_fd, write_fd = os.pipe()

    env = os.environ.copy()
    env["PRED_MUTPRI_RESULT_FD"] = str(write_fd)

    proc = None
    try:
        proc = subprocess.Popen(
            [
                bash_exe,
                str(af3_script),
                "-PDB",
                pdb,
                "-CHAIN",
                chain,
                "-Mut",
                mut,
            ],
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            pass_fds=(write_fd,),
        )

        # Parent must close its copy of the write end so EOF is observable.
        os.close(write_fd)
        write_fd = -1

        _, stderr_text = proc.communicate()

        chunks = []
        while True:
            block = os.read(read_fd, 4096)
            if not block:
                break
            chunks.append(block)

        payload = b"".join(chunks).decode(
            "ascii",
            errors="strict",
        ).strip()

        if proc.returncode != 0:
            raise RuntimeError(
                "AF3 stage failed.\n"
                + (stderr_text or "").strip()
            )

        if not payload:
            raise RuntimeError(
                "AF3 stage returned no internal ES value."
            )

        value = float(payload)

        if not math.isfinite(value):
            raise ValueError(
                f"AF3 ES is not finite: {value}"
            )

        return value

    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass

        if write_fd >= 0:
            try:
                os.close(write_fd)
            except OSError:
                pass


def cleanup_generated_mutant(
    mutant_path: Path | None,
    existed_before: bool,
):
    if (
        mutant_path is not None
        and not existed_before
        and mutant_path.exists()
    ):
        try:
            mutant_path.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pred-MutPRI prediction: "
            "38 internal features -> XGBoost ΔΔG."
        )
    )
    ap.add_argument("-PDB", required=True)
    ap.add_argument("-CHAIN", required=True)
    ap.add_argument("-Mut", required=True)
    ap.add_argument(
        "--model_json",
        required=True,
        help="Path to the XGBoost Booster JSON model.",
    )
    ap.add_argument(
        "--bash",
        default="bash",
    )
    ap.add_argument(
        "--protinter",
        default="protinter",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    pdb = args.PDB.strip()
    chain = args.CHAIN.strip()
    mut = args.Mut.strip()

    model_json = Path(
        args.model_json
    ).expanduser()

    if not model_json.is_absolute():
        model_json = (
            script_dir / model_json
        ).resolve()
    else:
        model_json = model_json.resolve()

    if not model_json.exists():
        raise FileNotFoundError(
            f"Model not found: {model_json}"
        )

    wt_pdb = script_dir / f"{pdb}.pdb"
    if not wt_pdb.exists():
        raise FileNotFoundError(
            f"Input PDB not found: {wt_pdb}"
        )

    expected_mutant = script_dir / f"{pdb}_1.pdb"
    mutant_existed_before = expected_mutant.exists()
    mutant_path = None

    try:
        # 1) Six structural features stay in Python memory.
        struct_features, mutant_path = quiet_call(
            compute_structural_features,
            pdb,
            chain,
            mut,
            protinter_exe=args.protinter,
            script_dir=script_dir,
        )

        if len(struct_features) != 6:
            raise RuntimeError(
                "Internal structural feature count error."
            )

        # 2) Thirty network features stay in Python memory.
        network_features = quiet_call(
            compute_network_features,
            pdb,
            chain,
            mut,
            script_dir=script_dir,
            wt_pdb=wt_pdb,
            mut_pdb=mutant_path,
        )

        if len(network_features) != 30:
            raise RuntimeError(
                "Internal network feature count error."
            )

        # Mutant FoldX PDB is no longer needed after network extraction.
        cleanup_generated_mutant(
            mutant_path,
            mutant_existed_before,
        )
        mutant_path = None

        # 3) AF3 ES crosses environments through an anonymous pipe.
        af3_script = script_dir / "run_af3_all.sh"
        if not af3_script.exists():
            raise FileNotFoundError(
                f"Missing {af3_script}"
            )

        es_value = run_af3_es_via_pipe(
            args.bash,
            af3_script,
            pdb,
            chain,
            mut,
            cwd=script_dir,
        )

        # 4) ESM scalar stays in Python memory.
        esm_value = quiet_call(
            compute_esm_feature,
            pdb,
            chain,
            mut,
            script_dir=script_dir,
        )

        features = np.asarray(
            struct_features
            + network_features
            + [float(es_value)]
            + [float(esm_value)],
            dtype=float,
        )

        if features.size != 38:
            raise RuntimeError(
                f"Expected 38 features, got {features.size}"
            )

        if not np.all(np.isfinite(features)):
            raise ValueError(
                "At least one internal feature is non-finite."
            )

        booster = load_xgb_booster_json(
            model_json
        )

        import xgboost as xgb

        dmat = xgb.DMatrix(
            features.reshape(1, -1)
        )
        pred = booster.predict(dmat)
        y0 = float(
            np.asarray(pred).ravel()[0]
        )

        # Best-effort cleanup of the explicit NumPy feature buffer.
        # This is hygiene, not a cryptographic guarantee: Python/NumPy/XGBoost
        # may have made internal copies that cannot be reliably zeroized here.
        try:
            features.fill(0.0)
        except Exception:
            pass

        for seq in (struct_features, network_features):
            try:
                for i in range(len(seq)):
                    seq[i] = 0.0
            except Exception:
                pass

        es_value = 0.0
        esm_value = 0.0

        # The only normal stdout payload.
        print(f"{y0:.6f}")

    finally:
        cleanup_generated_mutant(
            mutant_path,
            mutant_existed_before,
        )


if __name__ == "__main__":
    main()
