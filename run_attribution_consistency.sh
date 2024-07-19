#$ -S /bin/bash
#$ -cwd
#$ -N attribution_consistency
#$ -l m_mem_free=80G
#$ -o stdout.attribution_consistency.out
#$ -e stderr.attribution_consistency.out


python run_attribution_consistency.py
