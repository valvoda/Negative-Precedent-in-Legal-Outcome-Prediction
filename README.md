# Negative Precedent in Legal Outcome Prediction
This is a repository for code used in the paper: On the Role of Negative Precedent in Legal Outcome Prediction

## Model options
- baseline_positive
- baseline_negative
- mtl
- claim_outcome
- joint_model
- claims

To train the Claim-Outcome model, first train a baseline_positive and claims model and provide a path to them using the --pos_path and --claim_path arguments. You must also set the --inference flag.

## Get Started
Create a conda environment with the envirionment.yml file.

## Outcome corpus
- Outcome corpus: https://drive.google.com/file/d/1znbSf0vLJD-CxqpyzslxFw-vEe4qXOxw/view?usp=sharing
- Chalkidis et. al. corpus: https://drive.google.com/file/d/11ZvQf--QPb6Ut78YuNDq0sZNBVdI36by/view?usp=sharing
