from transformers import BertForSequenceClassification

def BERT_sentence(name: str, num_labels: int):
    model = BertForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model



def get_model(name, save_name, *args):
    if save_name in ['BERT_sentence_CL', 'BERT_sentence_IT']:
        num_label = args[0]
        return BERT_sentence(name, num_label)
