name=1000_n4_simple_uncertain
n_avg=[4]

#!/bin/sh
#BSUB -J 1000_n4_simple_uncertain
#BSUB -o outputs/1000_n4_simple_uncertain/%J.out
#BSUB -e outputs/1000_n4_simple_uncertain/%J.err
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
python src/main_ESC50.py hydra.run.dir=$dir n_avg=$n_avg
