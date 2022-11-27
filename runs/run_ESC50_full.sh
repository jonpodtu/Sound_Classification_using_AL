name=full_MobileNet

#!/bin/sh
#BSUB -J full_MobileNet
#BSUB -o outputs/full_MobileNet/%J.out
#BSUB -e outputs/full_MobileNet/%J.err
#BSUB -q gpua100
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
python src/main_ESC50.py hydra.run.dir=$dir
