# RR-FSL
Rhetorical Role Labeling using Few-Shot Learning Techniques
-----------------------------------------------------------

- This repo is for code developed during TUM practrical course: Legal NLP practical lab @2023



## Experiments Catalogue
for more detailed description about each experiment, check experiments_catalogue.txt

| Path                                            | Description                                                                                                                                           | CL on CL                                  | CL on IT        | IT on IT        | IT on CL        |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------|-----------------|-----------------|
| code/baseline_BERT_only                         | naive baseline model. model is bert-base-uncased with feed forward layer on top for classification. BERT is not forzen. BERT generate embeddings.     | 0.56                                      | 0.42            | 0.62            | 0.54            |
| code/baseline_hierarichal_BiLSTM/Hierarichal... | transform the documents of sentences into documents of embeddings using BERT bert-base-uncased.                                                     | 0.62                                      | -               | -               | -               |
| code/baseline_hierarichal_BiLSTM/Hierarichal... | transform the documents of sentences into documents of embeddings using BERT bert-base-uncased.                                                     | 0.39                                      | -               | -               | -               |
| code/copied/hier-bilstm-crf-baseline.ipynb      | the experiment is from [here](https://github.com/Exploration-Lab/Rhetorical-Roles/blob/main/Code/models/hier-bilstm-crf-baseline.ipynb).                | 0.38                                      | -               | -               | -               |
| code/prototypical/prototypical-CL.ipynb         | a prototypical network with LSTM encoder over the BERT embeddings.                                                                                    | F1-score for k=2: 49.23% (+-10.41%)<br>F1-score for k=4: 51.08% (+-9.09%)<br>F1-score for k=8: 51.74% (+-8.35%)<br>F1-score for k=16: 52.54% (+-7.76%)<br>F1-score for k=32: 53.31% (+-7.41%)| -               | -               | -               |

