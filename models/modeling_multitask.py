
import warnings

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel,BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertModel,BertPredictionHeadTransform
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput, \
    QuestionAnsweringModelOutput, TokenClassifierOutput

from models.fusion_embedding import FusionBertEmbeddings
from models.modeling_glycebert import GlyceBertModel
from datasets.utils import Pinyin

SMALL_CONST = 1e-15

class Phonetic_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pinyin=Pinyin()
        self.transform = BertPredictionHeadTransform(config)
        self.sm_classifier=nn.Linear(config.hidden_size,self.pinyin.sm_size)
        self.ym_classifier=nn.Linear(config.hidden_size,self.pinyin.ym_size)
        self.sd_classifier=nn.Linear(config.hidden_size,self.pinyin.sd_size)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sm_scores = self.sm_classifier(sequence_output)
        ym_scores = self.ym_classifier(sequence_output)
        sd_scores = self.sd_classifier(sequence_output)
        return sm_scores,ym_scores,sd_scores

class Pinyin_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.classifier=nn.Linear(config.hidden_size, 1378)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        scores = self.classifier(sequence_output)
        return scores

class MultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Phonetic_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        sm_scores,ym_scores,sd_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, sm_scores,ym_scores,sd_scores

class AblationHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Phonetic_relationship = Pinyin_Classifier(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        pinyin_scores = self.Phonetic_relationship(sequence_output)
        return prediction_scores, pinyin_scores

class GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        pinyin_labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gamma=1,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0)*(input_ids != 101)*(input_ids != 102).long()
        outputs = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores, sm_scores,ym_scores,sd_scores = self.cls(sequence_output)

        masked_lm_loss = None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)

        phonetic_loss=None
        if pinyin_labels is not None:
            active_loss = loss_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, pinyin_labels[...,0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[...,1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)
            active_labels = torch.where(
                active_loss, pinyin_labels[...,2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss=(sm_loss+ym_loss+sd_loss)/3

        loss=None
        if masked_lm_loss is not None :
            loss=masked_lm_loss 
            if phonetic_loss is not None:
                loss+= phonetic_loss *gamma
        
        if not return_dict:
            output = (prediction_scores, sm_scores,ym_scores,sd_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_a,
        hidden_states_b,
        attention_mask=None,
        output_attentions=False,
    ):
        query_layer = self.query(hidden_states_a)
        key_layer = self.transpose_for_scores(self.key(hidden_states_b))
        value_layer = self.transpose_for_scores(self.value(hidden_states_b))

        query_layer = self.transpose_for_scores(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = context_layer

        return outputs



class Dynamic_GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(Dynamic_GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)
        self.loss_fct = CrossEntropyLoss(reduction= 'none')

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tgt_pinyin_ids=None,
        pinyin_labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gamma=1,
        var=1,
        **kwargs
    ):

        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0)*(input_ids != 101)*(input_ids != 102).long()
        outputs_x = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_x = outputs_x[0]
        if tgt_pinyin_ids is not None:
            with torch.no_grad():
                outputs_y = self.bert(
                    labels,
                    tgt_pinyin_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoded_y = outputs_y[0]
                pron_x = self.cls.Phonetic_relationship.transform(encoded_x)
                pron_y = self.cls.Phonetic_relationship.transform(encoded_y) #[bs, seq, hidden_states]
                assert pron_x.shape == pron_y.shape
                sim_xy = F.cosine_similarity(pron_x, pron_y, dim= -1) #[ns, seq]
                factor = torch.exp( -((sim_xy -1.0) / var).pow(2)).detach()


        prediction_scores, sm_scores,ym_scores,sd_scores = self.cls(encoded_x)

        
        masked_lm_loss = None
        phonetic_loss=None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None and pinyin_labels is not None:

            active_loss = loss_mask.view(-1) == 1

            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[...,0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[...,1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[...,2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss=(sm_loss+ym_loss+sd_loss)/3

            def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)

            masked_lm_loss = weighted_mean(torch.ones_like(masked_lm_loss), masked_lm_loss)
            phonetic_loss = weighted_mean(factor.view(-1), phonetic_loss)


        loss=None
        if masked_lm_loss is not None and phonetic_loss is not None:
            loss=masked_lm_loss 
            loss+= phonetic_loss *gamma

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs_x.hidden_states,
            attentions=outputs_x.attentions,
        )

