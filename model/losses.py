import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import utils.distributed_utils as utils
import torch.distributed.nn
from model.box_utils import get_matched_indices
import configs.cc_config as config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed, logit_scale):
        # image_embed = outputs['image_embed']
        # text_embed = outputs['text_embed']
        # logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather with gradient
        image_embed_all = torch.cat(torch.distributed.nn.all_gather(image_embed), dim=0)
        text_embed_all = torch.cat(torch.distributed.nn.all_gather(text_embed), dim=0)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss.clone().detach(), 'clip_acc': acc}

class WordContrastiveLoss(nn.Module):
    def __init(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
    
    def forward(self,model_out, desc_features, obj_embeds,criterion, obj_boxes, num_queries, avail_idx, logit_scale):
        batch_desc_embeds = F.normalize(desc_features,dim=-1)
        pred_desc_embeds = F.normalize(obj_embeds,dim=-1)
        #print('Batch:',batch_desc_embeds.shape, 'Preds',pred_desc_embeds.shape)
        indices = get_matched_indices('all_boxes',
                                            criterion,
                                            model_out,
                                            obj_boxes,
                                            None,
                                            None,
                                            n_queries=num_queries)
        preds_selected = []
        gt_selected_indices = []
        for i, (src,tgt) in enumerate(indices):
            #src and tgt are of the form tuples of indices to be selected
            preds_selected.append(pred_desc_embeds[i,src])
            gt_selected_indices.extend([int(sum(avail_idx[:i*config.max_boxes]) + t) for t in tgt])
        preds_selected = torch.cat(preds_selected,dim=0)
        #print('Preds selected:',preds_selected.shape)

        sim_desc_matrix = logit_scale * torch.matmul(preds_selected, batch_desc_embeds.T)
        labels_desc = torch.tensor(gt_selected_indices,device=sim_desc_matrix.device)
        loss = F.cross_entropy(sim_desc_matrix, labels_desc)
        with torch.no_grad():
            acc = 100*(torch.argmax(sim_desc_matrix,dim=1) == labels_desc).float().mean()
        
        return {'word_loss': loss, 'word_acc': acc}
