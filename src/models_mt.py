# Author: Atharva Kulkarni
# Base code taken from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py

import math
import numpy as np
import os
from tqdm import tqdm
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoModelForImageClassification, 
    AdamW, 
    get_scheduler,
    AutoConfig
)

from typing import List, Optional, Tuple, Union
from  transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertOnlyMLMHead,
)

from  transformers.models.vit.modeling_vit import (
    ViTPreTrainedModel,
    ViTModel,
    ViTConfig,
)


# ----------------------------------------- Optimizers -----------------------------------------

def get_bert_optim(network, lr, weight_decay):
    print("\nUsing AdamW Optimizer\n")
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8
    )
    return optimizer
    

def get_sgd_optim(network, lr, weight_decay):
    print("\nUsing SGD Optimizer\n")
    return torch.optim.SGD(
        network.parameters(),
        lr=lr,
        momentum=0.9
    )

# ----------------------------------------- BERT Classifier Models -----------------------------------------

class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        
 
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return [logits, outputs[0]]



class ViTClassifier(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else torch.nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])
        return [logits, outputs[0]]
    
        
# ----------------------------------------- Wrapper Models -----------------------------------------

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2])


class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, interpolate_pos_encoding=True)
    

# ----------------------------------------- ERM Model -----------------------------------------

class ERM(torch.nn.Module):
    def __init__(self, hparams, dataloader):
        super().__init__()
        self.hparams = hparams
        dataset = dataloader.dataset
        self.n_batches = len(dataloader)
        self.data_type = dataset.data_type
        self.n_classes = len(set(dataset.y))
        self.n_groups = len(set(dataset.g))
        self.n_examples = len(dataset)
        self.last_epoch = 0
        self.best_selec_val = 0
        self.init_model_()
        print("\nUsing ERM Model...\n")
        self.l1 = False
        if "l1" in hparams.method:
            self.l1 = True
            self.l1_normalizer = None
            print("\nUsing L1 penalty on output embeddings...\n\n")

        
    def init_model_(self):

        if self.data_type == "text":
            optim="adamw"
            
            if self.hparams.model_name_or_path is None:
                self.hparams.model_name_or_path = 'bert-base-uncased'     
            
            if os.path.isdir(self.hparams.model_name_or_path):
                print(f"\nLoading BERT previous checkpoint from {self.hparams.model_name_or_path}\n")
            else:
                print(f"\nLoading BERT {self.hparams.model_name_or_path} model\n")
            self.network = BertWrapper(
                BertClassifier.from_pretrained(
                    self.hparams.model_name_or_path, 
                    num_labels=self.n_classes
                )
            )
            
        elif self.data_type == "images":  
            optim="sgd"
            
            if self.hparams.model_name_or_path is None:
                self.hparams.model_name_or_path = 'google/vit-base-patch16-224-in21k'
                
            else:
                if os.path.isdir(self.hparams.model_name_or_path):
                    print(f"\nLoading ViT previous checkpoint from {self.hparams.model_name_or_path}\n")
                else:
                    print(f"\nLoading ViT {self.hparams.model_name_or_path} model\n")

                self.network = ViTWrapper(
                    ViTClassifier.from_pretrained(
                        self.hparams.model_name_or_path, 
                        num_labels=self.n_classes,
                    )
                )
                
        self.network.zero_grad()
        
        self.clip_grad = optim == "adamw"
        optimizers = {
            "adamw": get_bert_optim,
            "sgd": get_sgd_optim
        }

        self.optimizer = optimizers[optim](
            self.network,
            self.hparams.learning_rate,
            self.hparams.weight_decay
        )
        
        if self.hparams.max_train_steps is None:
            self.hparams.max_train_steps = self.hparams.num_train_epochs * self.n_batches
        
        if self.hparams.lr_scheduler_type == "cosine_with_restarts":
            print("\nusing cosine learning rate scheduler...\n")
            
            if self.hparams.num_warmup_steps is None:
                self.hparams.num_warmup_steps = 500
                
            self.lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.max_train_steps
            )
            
        else:
            print("\nusing linear rate scheduler...\n")
            self.lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams.max_train_steps
            )

        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.cuda()


    def compute_loss_value_(self, i, x, y, g, epoch):
        if self.l1:
            l1_loss = torch.tensor(0.0, requires_grad=True)
            logits, output_embed = self.network(x)
            if self.l1_normalizer is None:
                self.l1_normalizer = output_embed.numel()
            l1_loss = torch.norm(output_embed, p=1)/self.l1_normalizer
            erm_loss = self.loss(logits, y).mean()
            return [erm_loss ,l1_loss]
        else:
            logits, _ = self.network(x)
            erm_loss = self.loss(logits, y).mean()
            return erm_loss


    def update(self, index, i, x, y, g, epoch):
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        if self.l1:
            erm_loss, l1_loss = self.compute_loss_value_(i, x, y, g, epoch)
            loss_value = (self.hparams.erm_weight * erm_loss) + (self.hparams.reg_weight * l1_loss)
        else:
            loss_value = self.compute_loss_value_(i, x, y, g, epoch)

        if loss_value is not None:
            loss_value.backward()
            loss_value = loss_value.item()
                
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            self.optimizer.zero_grad()
            self.network.zero_grad()
        else:
            print('\n\nAlert -- encountered NaN loss -- \n\n')
            loss_value = torch.tensor(0.0, requires_grad=True)
            loss_value.item()
            
        self.last_epoch = epoch
        if self.l1:
            return erm_loss.detach().float(), l1_loss.detach().float() 
        else:
            return loss_value


    def predict(self, x):
        return self.network(x)[0]


    def accuracy(self, loader, data_type):
        nb_groups = loader.dataset.nb_groups
        nb_labels = loader.dataset.nb_labels
        corrects = torch.zeros(nb_groups * nb_labels)
        totals = torch.zeros(nb_groups * nb_labels)
        self.eval()

        preds = []
        gold = []
        group_labels = []

        with torch.no_grad():
            inference_loss = 0.0
            for i, x, y, g in tqdm(loader, desc=f"{data_type} Inference Iteration"):
                predictions = self.predict(x.cuda())
                
                if data_type != 'tr':
                    inference_loss += self.loss(predictions, y.cuda()).mean().item()
                
                if predictions.squeeze().ndim == 1:
                    predictions = (predictions > 0).cpu().eq(y).float()
                else:
                    predictions = predictions.argmax(1).cpu().eq(y).float()
                    
                groups = (nb_groups * y + g)
                for gi in groups.unique():
                    corrects[gi] += predictions[groups == gi].sum()
                    totals[gi] += (groups == gi).sum()

                preds.extend(predictions)
                gold.extend(y)
                group_labels.extend(g)

        indices = np.where(totals == 0.0)[0]
        totals = np.delete(totals, indices)
        corrects = np.delete(corrects, indices)
        
        corrects, totals = corrects.tolist(), totals.tolist()
        print(f"\nindices: {indices}\n")
        print(f"\ncorrects: {corrects}\n")
        print(f"\ntotals: {totals}\n")
        print(f"\nsum(corrects) / sum(totals): {sum(corrects) / sum(totals)}\n")
        self.train()

        if data_type != 'tr':
            return sum(corrects) / sum(totals),\
                [c/t for c, t in zip(corrects, totals)], inference_loss/len(loader), \
                [int(x.item()) for x in preds], \
                [int(x.item()) for x in gold], \
                [int(x.item()) for x in group_labels]
        else:
            return sum(corrects) / sum(totals),\
                [c/t for c, t in zip(corrects, totals)], \
                [int(x.item()) for x in preds], \
                [int(x.item()) for x in gold], \
                [int(x.item()) for x in group_labels]


    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]
        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])

            
    def save(self, fname):
        lr_dict = None
        if self.lr_scheduler is not None:
            lr_dict = self.lr_scheduler.state_dict()
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val,
            },
            fname,
        )



# ----------------------------------------- groupDRO Model -----------------------------------------

class GroupDRO(ERM):
    def __init__(self, hparams, dataset):
        super(GroupDRO, self).__init__(hparams, dataset)
        self.register_buffer(
            "q", torch.ones(self.n_classes * self.n_groups).cuda())
        print("\nUsing groupDRO Model...\n")


    def groups_(self, y, g):
        idx_g, idx_b = [], []
        all_g = y * self.n_groups + g
        for g in all_g.unique():
            idx_g.append(g)
            idx_b.append(all_g == g)

        return zip(idx_g, idx_b)


    def compute_loss_value_(self, i, x, y, g, epoch):
        logits, _ = self.network(x)
        losses = self.loss(logits, y)
        for idx_g, idx_b in self.groups_(y, g):
            self.q[idx_g] *= (
                0.1 * losses[idx_b].mean()).exp().item()
        self.q /= self.q.sum()
        loss_value = 0
        for idx_g, idx_b in self.groups_(y, g):
            loss_value += self.q[idx_g] * losses[idx_b].mean()
        return loss_value



# ----------------------------------------- JTT Model -----------------------------------------

class JTT(ERM):
    def __init__(self, hparams, dataset):
        super(JTT, self).__init__(hparams, dataset)
        self.register_buffer(
            "weights", torch.ones(self.n_examples, dtype=torch.long))
        print("\nUsing JTT Model...\n")

    def compute_loss_value_(self, i, x, y, g, epoch):
        if epoch == self.hparams.T + 1 and\
           self.last_epoch == self.hparams.T:
            self.init_model_()

        predictions, _ = self.network(x)
        predictions = predictions.cuda()

        if epoch != self.hparams.T:
            loss_value = self.loss(predictions, y).mean()
        else:
            self.eval()
            if predictions.squeeze().ndim == 1:
                # wrong_predictions = (predictions > 0).cpu().ne(y).float()
                wrong_predictions = (predictions > 0).cpu().ne(y.cpu()).int()

            else:
                # wrong_predictions = predictions.argmax(1).cpu().ne(y).float()
                wrong_predictions = predictions.argmax(1).cpu().ne(y.cpu()).int()

            self.weights[i] += wrong_predictions.detach() * (self.hparams.up - 1)
            self.train()
            loss_value = None

        return loss_value


    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]

        if self.last_epoch > self.hparams["T"]:
            self.init_model_()

        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])



# ----------------------------------------- MTBERT Model -----------------------------------------            

class MLMBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        
        # ERM part 
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        
        # MLM part
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        
        # SimCSE part
        self.cache = dict()
        self.temperature = 1.0
        
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        
        
    def forward(
        self,
        task_name = 'erm',
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if task_name == 'erm':
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            self.cache['pooled_output'] = outputs[1].detach().clone() # Cache outputs for SimCLR part
            return [logits, outputs[0]]

        else:
            if task_name in ["erm_mt", "erm_mt_l1"]:
                sequence_output = outputs[0]
                prediction_scores = self.cls(sequence_output)
            
                mlm_loss = None
                if labels is not None:
                    loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
                    mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

                return [mlm_loss, outputs[0]]
                
            elif task_name in ["erm_mt2", "erm_mt2_l1"]:
                sequence_output = outputs[0]
                prediction_scores = self.cls(sequence_output)
            
                clm_loss = None
                if labels is not None:
                    # we are doing next-token prediction; shift prediction scores and input ids by one
                    shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
                    labels = labels[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    clm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                    return [clm_loss, outputs[0]]
                    

# ----------------------------------------- MTBERT Wrapper Model -----------------------------------------            

class MTBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, task_name, x):
        if task_name == 'erm':
            forward_kwargs = {
                'task_name': task_name,
                'input_ids': x[:, :, 0],
                'attention_mask': x[:, :, 1],
                'token_type_ids': x[:, :, 2]
            } 
            self.model.config.update(
                {
                    "is_decoder": False,
                }
            )
            
        
        elif task_name in ['erm_mt', 'erm_mt_l1', 'erm_mt2', 'erm_mt2_l1']:
            forward_kwargs = {
                'task_name': task_name,
                'input_ids': x['input_ids'],
                'attention_mask': x['attention_mask'],
                'token_type_ids': x['token_type_ids'],
                'labels': x['labels']
            } 
            if task_name in ['erm_mt2', 'erm_mt2_l1']:
                self.model.config.update(
                    {
                        "is_decoder": True,
                    }
                )
                
        result = self.model(**forward_kwargs)
        return result 

    
      
# ----------------------------------------- MTViTModel SimCLR -----------------------------------------            

class SimCLRViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
        self.config.update(
            {
                "image_size": 384,
            }
        )

        # ERM part 
        self.classifier = torch.nn.Linear(
            config.hidden_size, config.num_labels
        ) if config.num_labels > 0 else torch.nn.Identity()

        # SimCLR part
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU()
        )
        self.temperature = 1.0
        
        # Initialize weights and apply final processing
        self.post_init()
        
        self.cache = dict()
        
    
    def forward(
        self,
        task_name = 'erm',
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if task_name == 'erm':
            outputs = self.vit(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            pooled_output = sequence_output[:, 0, :]
            logits = self.classifier(pooled_output)
            self.cache['pooled_output'] = pooled_output.detach().clone() # Cache outputs for SimCLR part
            return [logits, sequence_output]

        else:
            cache_pooled_output = features = self.cache.pop('pooled_output')
            device = cache_pooled_output.get_device()
            proj1 = self.projection_head(cache_pooled_output)
            
            outputs = self.vit(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            sequence_output2 = outputs[0]
            proj2 = self.projection_head(sequence_output2[:, 0, :])

            features = torch.cat([proj1, proj2], dim=0)
            features = torch.nn.functional.normalize(features, dim=1)
            similarity_matrix = torch.matmul(features, features.T)
            
            batch_size = cache_pooled_output.shape[0]
            labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(device)
                        
            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            # assert similarity_matrix.shape == labels.shape

            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            logits = logits / self.temperature
                        
            self.loss_fct = torch.nn.CrossEntropyLoss().to(self.device)
            nt_xnet_loss = self.loss_fct(logits, labels)
            return [nt_xnet_loss, outputs[0]]      
            
            
            
# ----------------------------------------- MTViTModel Wrapper -----------------------------------------            
 
class MTViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, task_name, x):
        if task_name == 'erm':
            forward_kwargs = {
                'task_name': task_name,
                'pixel_values': x,
                'interpolate_pos_encoding': True
            } 
        
        elif task_name in ['erm_mt2', 'erm_mt2_l1']:
            forward_kwargs = {
                'task_name': task_name,
                'pixel_values': x['pixel_values'],
                'interpolate_pos_encoding': True
            } 
        elif task_name in ['erm_mt', 'erm_mt_l1']:
            forward_kwargs = {
                'task_name': task_name,
                'pixel_values': x['pixel_values'],
                'bool_masked_pos': x['bool_masked_pos'],
                'interpolate_pos_encoding': True
            } 

        return self.model(**forward_kwargs)  
    
    

# ----------------------------------------- ERM_MT Model -----------------------------------------            

class ERM_MT(torch.nn.Module):
    def __init__(self, hparams, dataloader):
        super().__init__()
        print("\nUsing ERM_MT Model...\n")
        self.hparams = hparams
        dataset = dataloader.dataset
        self.n_batches = len(dataloader)
        self.data_type = dataset.data_type
        self.n_classes = len(set(dataset.y))
        self.n_groups = len(set(dataset.g))
        self.n_examples = len(dataset)
        self.last_epoch = 0
        self.best_selec_val = 0
        self.l1 = False
        if "l1" in hparams.method:
            self.l1 = True
            self.erm_l1_normalizer = None
            self.mt_l1_normalizer = None
            print("\nUsing L1 penalty on output embeddings...\n\n")
        self.init_model_()
        
        
    def init_model_(self):
    
        if self.data_type == "text":
            optim = "adamw"
            if self.hparams.model_name_or_path is None:
                    self.hparams.model_name_or_path = 'bert-base-uncased'   
          
            elif self.hparams.method in ["erm_mt2", "erm_mt2_l1"]:
                if os.path.isdir(self.hparams.model_name_or_path):
                    print(f"\nLoading previous CLM multitask checkpoint from {self.hparams.model_name_or_path}\n")
                else:
                    print(f"\nLoading CLM multitask {self.hparams.model_name_or_path} model\n")
                self.network = MTBertWrapper(
                    MLMBertModel.from_pretrained(
                        self.hparams.model_name_or_path, 
                        num_labels=self.n_classes,
                        is_decoder=True
                    )
                )
                
            else:
                if os.path.isdir(self.hparams.model_name_or_path):
                    print(f"\nLoading previous MLM multitask checkpoint from {self.hparams.model_name_or_path}\n")
                else:
                    print(f"\nLoading MLM multitask {self.hparams.model_name_or_path} model\n")
                self.network = MTBertWrapper(
                    MLMBertModel.from_pretrained(
                        self.hparams.model_name_or_path, 
                        num_labels=self.n_classes
                    )
                )
            
        elif self.data_type == "images":   
            optim = "sgd"
            
            if self.hparams.model_name_or_path is None:
                    self.hparams.model_name_or_path = 'google/vit-base-patch16-224-in21k'
                    
            if self.hparams.method in ["erm_mt2", "erm_mt2_l1"]:
                if os.path.isdir(self.hparams.model_name_or_path):
                    print(f"\nLoading previous SimCLR multitask checkpoint from {self.hparams.model_name_or_path}\n")
                else:
                    print(f"\nLoading SimCLR multitask {self.hparams.model_name_or_path} model\n")
                self.network = MTViTWrapper(
                    SimCLRViTModel.from_pretrained(
                        self.hparams.model_name_or_path, 
                        num_labels=self.n_classes
                    )
                )
            else:
                if os.path.isdir(self.hparams.model_name_or_path):
                    print(f"\nLoading previous MIM multitask checkpoint from {self.hparams.model_name_or_path}\n")
                else:
                    print(f"\nLoading MIM multitask {self.hparams.model_name_or_path} model\n")
                self.network = MTViTWrapper(
                    MIMViTModel.from_pretrained(
                        self.hparams.model_name_or_path, 
                        num_labels=self.n_classes
                    )
                )
            
        self.network.zero_grad()
        
        self.clip_grad = optim == "adamw"
        optimizers = {
            "adamw": get_bert_optim,
            "sgd": get_sgd_optim
        }

        self.optimizer = optimizers[optim](
            self.network,
            self.hparams.learning_rate,
            self.hparams.weight_decay
        )
        
        if self.hparams.max_train_steps is None:
            self.hparams.max_train_steps = self.hparams.num_train_epochs * self.n_batches
        
        if self.hparams.lr_scheduler_type == "cosine_with_restarts":
            print("\nusing cosine learning rate scheduler...\n")
            if self.hparams.num_warmup_steps is None:
                self.hparams.num_warmup_steps = 500
                
        else:
            print("\nusing linear rate scheduler...\n")
            if self.hparams.num_warmup_steps is None:
                self.hparams.num_warmup_steps = 0.06 * self.hparams.max_train_steps
            
        self.lr_scheduler = get_scheduler(
            name=self.hparams.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.max_train_steps
        )
             
        self.erm_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self.mt_loss_fct = torch.nn.CrossEntropyLoss()
        self.cuda()



    def compute_loss_value_(self, i, x, y, g, epoch):
        erm_x, mt_x = x 
        erm_loss = torch.tensor(0.0, requires_grad=True)
        mt_loss = torch.tensor(0.0, requires_grad=True)
        erm_l1_loss = torch.tensor(0.0, requires_grad=True)
        mt_l1_loss = torch.tensor(0.0, requires_grad=True)
    
        if erm_x is not None:
            erm_x, y, g = erm_x.cuda(), y.cuda(), g.cuda()
            if self.l1:
                logits, output_embed = self.network('erm', erm_x)
                erm_loss = self.erm_loss_fct(logits, y).mean()
                if self.erm_l1_normalizer is None:
                    self.erm_l1_normalizer = output_embed.numel()
                erm_l1_loss = torch.norm(output_embed, p=1)/self.erm_l1_normalizer
            else:
                logits, _ = self.network('erm', erm_x)
                erm_loss = self.erm_loss_fct(logits, y).mean()
        
        if mt_x is not None:
            mt_x = {k:v.cuda() for k, v in mt_x.items()}
            
            if self.l1:
                mt_loss, output_embed = self.network(self.hparams.method, mt_x)
                if self.mt_l1_normalizer is None:
                    self.mt_l1_normalizer = output_embed.numel()
                mt_l1_loss = torch.norm(output_embed, p=1)/self.mt_l1_normalizer
            else:
                mt_loss, _ = self.network(self.hparams.method, mt_x)
                
        if self.l1:
            return [
                erm_loss/self.hparams.grad_acc, 
                mt_loss/self.hparams.mt_grad_acc, 
                (erm_l1_loss/self.hparams.grad_acc) + (mt_l1_loss/self.hparams.mt_grad_acc)
            ]
        else:  
            return [
                    erm_loss/self.hparams.grad_acc, 
                    mt_loss/self.hparams.mt_grad_acc
            ]



    def update(self, index, i, x, y, g, epoch, update_params=False): 
        if self.l1:
            erm_loss, mt_loss, l1_loss = self.compute_loss_value_(i, x, y, g, epoch)
            loss_value = (self.hparams.erm_weight * erm_loss) + (self.hparams.mt_weight * mt_loss) + (self.hparams.reg_weight * l1_loss)
        else:
            erm_loss, mt_loss = self.compute_loss_value_(i, x, y, g, epoch)
            loss_value = (self.hparams.erm_weight * erm_loss) + (self.hparams.mt_weight * mt_loss)
        if torch.isnan(loss_value):
            erm_loss = torch.tensor(0.0)
            mt_loss = torch.tensor(0.0) 
            l1_loss = torch.tensor(0.0) 
            print('\n\nAlert -- encountered NaN loss -- \n\n')
            del loss_value
        else:
            loss_value.backward()
        if update_params:
            self.step_params()               
        self.last_epoch = epoch
        if self.l1:
            return erm_loss.detach().float(), mt_loss.detach().float(), l1_loss.detach().float() 
        else:
            return erm_loss.detach().float(), mt_loss.detach().float()
    
    
    def step_params(self):
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.network.zero_grad()


    def predict(self, x):
        return self.network('erm', x)[0]


    def accuracy(self, loader, data_type):
        nb_groups = loader.dataset.nb_groups
        nb_labels = loader.dataset.nb_labels
        corrects = torch.zeros(nb_groups * nb_labels)
        totals = torch.zeros(nb_groups * nb_labels)
        self.eval()

        preds = []
        gold = []
        group_labels = []

        with torch.no_grad():
            inference_loss = 0.0
            for i, x, y, g in tqdm(loader, desc=f"{data_type} Inference Iteration"):
                predictions = self.predict(x.cuda())
                if data_type != 'tr':
                    inference_loss += self.erm_loss_fct(predictions, y.cuda()).mean().item()
                
                if predictions.squeeze().ndim == 1:
                    predictions = (predictions > 0).cpu().eq(y).float()
                else:
                    predictions = predictions.argmax(1).cpu().eq(y).float()
                groups = (nb_groups * y + g)
                for gi in groups.unique():
                    corrects[gi] += predictions[groups == gi].sum()
                    totals[gi] += (groups == gi).sum()

                preds.extend(predictions)
                gold.extend(y)
                group_labels.extend(g)

        indices = np.where(totals == 0.0)[0]
        totals = np.delete(totals, indices)
        corrects = np.delete(corrects, indices)
        
        corrects, totals = corrects.tolist(), totals.tolist()
        print(f"\nindices: {indices}\n")
        print(f"\ncorrects: {corrects}\n")
        print(f"\ntotals: {totals}\n")
        print(f"\nsum(corrects) / sum(totals): {sum(corrects) / sum(totals)}\n")
        self.train()

        if data_type != 'tr':
            return sum(corrects) / sum(totals),\
                [c/t for c, t in zip(corrects, totals)], inference_loss/len(loader), \
                [int(x.item()) for x in preds], \
                [int(x.item()) for x in gold], \
                [int(x.item()) for x in group_labels]
        else:
            return sum(corrects) / sum(totals),\
                [c/t for c, t in zip(corrects, totals)], \
                [int(x.item()) for x in preds], \
                [int(x.item()) for x in gold], \
                [int(x.item()) for x in group_labels]
                
                
    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]
        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])

    def save(self, fname):
        lr_dict = None
        if self.lr_scheduler is not None:
            lr_dict = self.lr_scheduler.state_dict()
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val,
            },
            fname,
        )

# ---------------------------------------------------------------------------------------------------------------------
