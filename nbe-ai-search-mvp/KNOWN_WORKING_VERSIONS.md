# Known Working Versions (tested 2026-07-22)

The ranges in requirements.txt are too loose and cause dependency conflicts
depending on install date (pip picks the newest available version, which may
break compatibility between torch/transformers/FlagEmbedding).

These EXACT versions were tested and confirmed working on Ubuntu 22.04, Python 3.11, CPU-only:

torch==2.4.1+cpu
transformers==4.44.2
FlagEmbedding==1.2.10
sentence-transformers==3.0.1
accelerate==0.34.2

Install order matters:
1. pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
2. pip install transformers==4.44.2
3. pip install FlagEmbedding==1.2.10
4. pip install sentence-transformers==3.0.1
5. pip install accelerate==0.34.2
6. pip install -r requirements.txt   (installs everything else; torch/transformers/etc. already satisfied)

For full reproducibility, see requirements.lock.txt (pip freeze output).
