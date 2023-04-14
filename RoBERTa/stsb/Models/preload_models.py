import sys, time, pickle, torch
from sentence_transformers import CrossEncoder

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
