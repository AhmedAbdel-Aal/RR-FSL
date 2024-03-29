import numpy as np
from models import BertHSLN

import torch

def save_npy(t, path):
  # Move the tensor from GPU to CPU
  result_cpu_tensor = t.cpu()

  # Convert the CPU tensor to a NumPy array
  result_numpy_array = result_cpu_tensor.numpy()

  # Save the NumPy array to an NPY file
  np.save(path, result_numpy_array)


def load_checkpoint(path, mconfig, device):
  model = BertHSLN(mconfig, num_labels = 7)
  model.load_state_dict(torch.load(path))
  model.to(device)
  return model




from allennlp.common.util import pad_sequence_to_length

def batch_to_tensor(b):
    # convert to dictionary of tensors and pad the tensors
    max_sentence_len = 128
    result = {}
    for k, v in b.items():

        if k in ["input_ids", "attention_mask"]:
            # determine the max sentence len in the batch
            max_sentence_len = -1
            for sentence in v:
                sentence = torch.cat(sentence)
                max_sentence_len = max(len(sentence), max_sentence_len)
            # pad the sentences to max sentence len
            for i, sentence in enumerate(v):
                v[i] = pad_sequence_to_length(sentence, desired_length=max_sentence_len)
        if k!='doc_name' and k!= 'label_ids':
            result[k] = torch.tensor(v).unsqueeze(0)
        elif k == 'label_ids':
            result[k] = torch.tensor(v)
        else:
            result[k] = v
    return result







