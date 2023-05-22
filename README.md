# RR-FSL
Rhetorical Role Labeling using Few-Shot Learning Techniques
-----------------------------------------------------------

- This repo is for code developed during TUM practrical course: Legal NLP practical lab @2023


## Repo structure
#### data:
The data used so far is from https://github.com/Exploration-Lab/Rhetorical-Roles. 

The data is in json format, run the ./data/Parse_data_to_csvs.ipynb to transform data into 6 csv files. 

The files are (train.csv, dev.csv, test.csv) for two different domains of Rhetorical role data (CL and IT).

#### code
files in this directory are the files needed to run an experiment. Assuming the experiments has name of <exp_name>, the needed files are:
  - <exp_name>_config.json
  - data.py
  - models.py
  - evaluation.py
  - utils.py
  - <exp_name>_colab.ipynb
  - <exp_name>_locally.ipynb

read the directory README.md for more information.
#### eval
results from evaluation scripts are saved here.

#### log
logs created during training.

#### models
where created models are saved.
