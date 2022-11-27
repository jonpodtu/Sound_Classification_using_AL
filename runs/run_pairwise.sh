#!/bin/sh
#BSUB -J pairwise
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -N
#BSUB -W 24:00
# end of BSUB options

# activate the virtual environment 
source venv/bin/activate

python src/get_pairwise.py