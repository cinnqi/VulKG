import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        print('bert_linear_size:{}'.format(self.num_labels))
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertLstmCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLstmCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, 1,
                             batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout1(sequence_output)
        sequence_output = self.dropout2(self.lstm(sequence_output)[0])
        logits = self.classifier(self.layernorm(sequence_output))
        outputs = (logits,)
        if labels is not None:
            # print("logits:",logits)
            # print("labels:",labels)
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs

        
class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores



