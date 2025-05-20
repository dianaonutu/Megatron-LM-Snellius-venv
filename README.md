# Megatron-LM on Snellius using virtual environment

This code base is for users to quickly set up a virtual environment for pretraining LLMs with Megatron-LM.

## Create virtual environment 
**Estimated time:** 10 minutes.
1. Load required modules. Note, this combination of modules together with PyTorch 2.6 and CUDA 12.6 is the best (e.g., fast flash attention installation).
```
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
```
2. Create a virtual environment inside a folder named `megatron`, which will contain both Megatron-LM and the environment.
`python -m venv megatron-venv`
3. Allocate compute node, activate virtual environment and install packages. **Estimated time:** 7 minutes
```
salloc -n 16 -t 30:00
source megatron-venv/bin/activate
./install.sh
```
4. Once finished, stop node allocation.
`exit`
