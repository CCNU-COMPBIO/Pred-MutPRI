To run the prediction pipeline, all `.py` and `.sh` scripts in the `scripts/` folder must be placed/run in the **same project root directory level** as external dependencies (e.g., `FoldX`, `mkdssp`/DSSP, `protinter`, `x3dna-dssr`).  
This is required because the scripts rely on **relative paths** to locate these tools and input files.
