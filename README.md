# KR_2025
## Instalation
1. Prepare proper version for torch+CUDA

   ``pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124``
   
2. Install additional requiremets

   ``pip install -r requiremets.txt``

3. Install SAM2 from source
   
   ``git clone https://github.com/facebookresearch/sam2.git /usr/local/sam2``
   
   ``pip install -e /usr/local/sam2 -q``

## Run benchmark
1. Load RefCOCO dataset

   ``python download_dataset.py``

2. Run benchmark

   ``python run_benchmark.py``

   
