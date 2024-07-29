#$ -S /bin/bash
#$ -cwd
#$ -N motif_test
#$ -l m_mem_free=80G 
#$ -o stdout.motifs.out
#$ -e stderr.motifs.out

python run_motifs.py