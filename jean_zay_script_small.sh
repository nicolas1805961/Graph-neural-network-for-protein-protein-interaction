#!/bin/bash
#SBATCH --job-name=gpu_mono          # nom du job
##SBATCH -C v100-16g                  # reserver des GPU 16 Go seulement
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
##SBATCH --qos=qos_gpu-dev            # qos_gpu-t4 qos_gpu-dev qos_gpu-t3
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
##SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
##SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
##SBATCH --hint=nomultithread         # hyperthreading desactive
##SBATCH --time=02:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00  
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/2.2.0

# echo des commandes lancees
set -x

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

#pip install pandas
#pip install biopython

# execution du code
python download_pdb.py