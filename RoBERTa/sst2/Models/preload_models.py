import sys, time, pickle, torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

class Roberta_Sentiment_Wrapper(torch.nn.Module):
    def __init__(self):
        super(Roberta_Sentiment_Wrapper, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english')

    def get_embeddings(self, input_ids):
        return self.model.roberta.embeddings(input_ids)

    def forward(self, embeddings):
        encoder_outputs = self.model.roberta.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        #         pooled_output = self.model.roberta.pooler(sequence_output) if self.model.roberta.pooler is not None else None
        #         roberta_outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

        logits = self.model.classifier(sequence_output)
        #         print(f'logits {logits} {logits.size()} {logits.dtype} {logits.requires_grad}')
        pred_prob = torch.softmax(logits, dim=1)[:, 1]
        # get the item at idx 1 because it corresponds to probability of being positive

        return pred_prob

def get_sst2_tok_n_model():
    tokenizer = RobertaTokenizerFast.from_pretrained('siebert/sentiment-roberta-large-english')
    model = Roberta_Sentiment_Wrapper()
    return tokenizer, model