## External dependencies (download links)

This pipeline relies on several external tools. Please install/download them first and make sure they are available as described below.

### FoldX
- Website: https://foldxsuite.crg.eu/
- Download/License: https://foldxsuite.crg.eu/academic-license-info

### DSSP / mkdssp
- DSSP (mkdssp) source and documentation: https://github.com/PDB-REDO/dssp
- Note: On some systems the executable may be named `mkdssp` instead of `dssp`.

### ProtInter
- Repository (ProtInter): https://github.com/maxibor/protinter

### DSSR (x3dna-dssr)
- 3DNA/DSSR homepage: http://x3dna.org/

### Cytoscape (for network features)
- Cytoscape homepage: https://cytoscape.org/
- Download page: https://cytoscape.org/download/
- Important: **Cytoscape must be running (opened) during prediction**, so that the pipeline can connect to it and compute network features.

### AlphaFold3 (for ES feature)
- AlphaFold3 (official): https://github.com/google-deepmind/alphafold3
- Note: Please follow the official installation and model-parameter download instructions.

## Environments

AlphaFold3 dependencies may conflict with ESM (and other Python packages used for prediction).  
Therefore, we created two separate conda environments:

- `af3`: used to ensure **AlphaFold3 runs correctly** (ES feature).
- `env1`: the main **prediction execution environment** (feature extraction + XGBoost prediction, including ESM-related steps).

## Python dependencies

Install the required Python packages in the `env1` environment (example using `pip`):

```bash
pip install numpy pandas scikit-learn xgboost biopython
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm
pip install py4cytoscape networkx
```

Notes:
- The exact PyTorch command depends on your CUDA version (or use CPU-only wheels).
- `py4cytoscape` requires Cytoscape to be running for the network feature step.

## Run prediction

From the project root directory, run:

```bash
./predict.sh -PDB 1A1T -CHAIN A -Mut E21K --model_json model.json
```

Replace the arguments with your own inputs:
- `-PDB`: PDB ID (e.g., `1A1T`)
- `-CHAIN`: chain ID (e.g., `A`)
- `-Mut`: mutation in the format `WT<pos>MUT` (e.g., `E21K`)
- `--model_json`: path to the trained XGBoost model file
