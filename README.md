# HPC Virtual environment Guide
1. Load module `module load python3/3.9.14`
2. `python3 -m venv venv`
3. `source venv/bin/activate`
4. Install requirements:  `pip install -r requirements.txt`
5. Install CUDA-compatible torch versions: `pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html`

# Bachelorproject
A repo for the bachelor project for the BSc in Artificial Intelligence and Data at the Technical University of Denmark, by Jonas Poulsen, Lasse Møller Sørensen and Christian Westerdahl. 