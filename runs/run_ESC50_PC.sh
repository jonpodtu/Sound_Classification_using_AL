name=PCA_VAAL
AL_model=["VAAL"]
model="outputs/PCA_VAAL/0_run/VAAL/MobileNet/models/12_mdl.pt"

#!/bin/sh
#BSUB -J PCA_VAAL
#BSUB -o outputs/PCA_VAAL/%J.out
#BSUB -e outputs/PCA_VAAL/%J.err
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -N
#BSUB -W 24:00
# end of BSUB options

nvidia-smi
# Load the cuda module
module load cuda/11.6

# activate the virtual environment 
source venv/bin/activate

dir=outputs/$name
CUDA_LAUNCH_BLOCKING=1 python src/main_ESC50.py hydra.run.dir=$dir AL_methods=$AL_model sample_first=$model
