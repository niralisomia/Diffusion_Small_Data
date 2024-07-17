#$ -S /bin/bash
#$ -cwd
#$ -N functional_similarity
#$ -l m_mem_free=80G 
#$ -o stdout.functional_sequence.out
#$ -e stderr.functional_sequence.out

python run_functional_similarity.py