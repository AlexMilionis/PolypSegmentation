#!/bin/bash
#SBATCH --job-name=my_training    # Όνομα εργασίας
#SBATCH --output=output_%j.log    # Αρχείο καταγραφής (το %j είναι το ID εργασίας)
#SBATCH --error=error_%j.log      # Αρχείο σφαλμάτων
source activate msc_thesis
python main.py exp1