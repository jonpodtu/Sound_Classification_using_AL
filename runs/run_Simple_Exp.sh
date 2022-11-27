init_budget=400

#!/bin/sh
#BSUB -J st_400
#BSUB -o outputs/st_400_%J.out
#BSUB -e outputs/st_400_%J.err
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

python src/main_ESC50.py initial_budget=$init_budget