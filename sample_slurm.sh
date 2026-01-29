#!/bin/bash
#SBATCH --job-name=crpropa # Job name
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<>   # Where to send mail	
#SBATCH --time=1-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --output=output/test/sample_slurm.log  # Standard output and error log
#SBATCH --mem=1G                    
#SBATCH --array=0-1
#SBATCH --partition=128x24

echo "Config file used:"
echo $1

module load crpropa 
export PYTHONPATH=$PYTHONPATH:/hb/home/mspletts/simCRPropaVoids/simCRpropa/:/hb/home/mspletts/fermiAnalysis/ 
export LSCRATCH="/hb/home/mspletts/simCRPropaVoids/simCRpropa/output/_scratch"
python simCRpropa/scripts/run_crpropa_em_cascade.py --conf $1 

