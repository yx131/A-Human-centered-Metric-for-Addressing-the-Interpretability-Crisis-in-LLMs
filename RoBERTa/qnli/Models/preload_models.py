import sys, time, pickle, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class roberta_qnli_wrapper(torch.nn.Module):
    def __init__(self):
        super(roberta_qnli_wrapper, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/qnli-distilroberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/qnli-distilroberta-base')

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


def get_qnli_tok_n_model():
    model = roberta_qnli_wrapper()
    tokenizer = model.get_tokenizer()
    return tokenizer, model
