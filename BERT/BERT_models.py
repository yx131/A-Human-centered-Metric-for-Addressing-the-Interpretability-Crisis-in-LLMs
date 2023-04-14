import sys, time, pickle
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
sys.path.insert(0, '../Utils')
from global_constants import gpu_device

class BERT_SST2_MODEL(torch.nn.Module):
    def __init__(self):
        super(BERT_SST2_MODEL, self).__init__()
        self.model_type = BertForSequenceClassification
        self.tokenizer_type = BertTokenizerFast
        self.model_name_str = 'Ghost1/bert-base-uncased-finetuned_for_sentiment_analysis1-sst2'
        
        self.model = self.model_type.from_pretrained(self.model_name_str).to(gpu_device)
        self.tokenizer = self.tokenizer_type.from_pretrained(self.model_name_str)
        
    def get_embeddings(self, input_ids):
        return self.model.bert.embeddings(input_ids)

    def forward(self, embeddings):
        encoder_outputs = self.model.bert.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)        
        logits = self.model.classifier(pooled_output)
#         print(f'logits {logits} {logits.size()} {logits.dtype} {logits.requires_grad}')
        pred_prob = torch.softmax(logits, dim=1)[:, 1]
#         get the item at idx 1 because it corresponds to probability of being positive

        return pred_prob    


class BERT_STSB_Model(torch.nn.Module):
    def __init__(self):
        super(BERT_QNLI_MODEL, self).__init__()
        self.model_type = BertForSequenceClassification
        self.tokenizer_type = BertTokenizerFast
        self.model_name_str = 'textattack/bert-base-uncased-QNLI'
        
        self.model = self.model_type.from_pretrained(self.model_name_str).to(gpu_device)
        self.tokenizer = self.tokenizer_type.from_pretrained(self.model_name_str)
        
    def get_embeddings(self, input_ids):
        return self.model.bert.embeddings(input_ids)

    def forward(self, embeddings):
        encoder_outputs = self.model.bert.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)        
        logits = self.model.classifier(pooled_output)
#         print(f'logits {logits} {logits.size()} {logits.dtype} {logits.requires_grad}')
        pred_prob = torch.softmax(logits, dim=1)[:, 1]
#         get the item at idx 1 because it corresponds to probability of being positive

        return pred_prob 

class BERT_QNLI_MODEL(torch.nn.Module):
    def __init__(self):
        super(BERT_QNLI_MODEL, self).__init__()
        self.model_type = BertForSequenceClassification
        self.tokenizer_type = BertTokenizerFast
        self.model_name_str = 'textattack/bert-base-uncased-QNLI'
        
        self.model = self.model_type.from_pretrained(self.model_name_str).to(gpu_device)
        self.tokenizer = self.tokenizer_type.from_pretrained(self.model_name_str)
        
    def get_embeddings(self, input_ids):
        return self.model.bert.embeddings(input_ids)

    def forward(self, embeddings):
        encoder_outputs = self.model.bert.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)        
        logits = self.model.classifier(pooled_output)
#         print(f'logits {logits} {logits.size()} {logits.dtype} {logits.requires_grad}')
        pred_prob = torch.softmax(logits, dim=1)[:, 1]
#         get the item at idx 1 because it corresponds to probability of being positive

        return pred_prob 



class roberta_stsb_wrapper(torch.nn.Module):
    def __init__(self):
        super(roberta_stsb_wrapper, self).__init__()
        self.cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')
        self.model = self.cross_encoder_model.model
        self.tokenizer = self.cross_encoder_model.tokenizer

    def get_embeddings(self, input_ids):
        return self.model.roberta.embeddings(input_ids)

    def forward(self, embeddings):
        outputs = self.model.roberta.encoder(embeddings)
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)
        pred_prob = torch.sigmoid(logits)
        # print(f'pred prob {pred_prob}')
        return pred_prob

    def get_tokenizer(self):
        return self.tokenizer


def get_stsb_tokenizer_n_model():
    model = roberta_stsb_wrapper()
    tokenizer = model.get_tokenizer()
    return tokenizer, model
