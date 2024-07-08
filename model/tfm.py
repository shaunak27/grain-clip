#Acknowledgements: Code adapted from DETR & HelpingHands
import copy
import math

# high level edits:
# removed positional encoding
# removed mask as it is 0 anyway
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, List
from timm.models.layers import  trunc_normal_

class Cross_Attention(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, hidden_dim=768,
                 return_intermediate_dec=False, sa_first=True,):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model) if normalize_before else None
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                sa_first = sa_first)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.enc_layers = num_encoder_layers

            
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src,query_embed):
        # flatten NxCxHxW to HWxNxC                       # in our case, 1xNxdim which is S,N,E i.e. the shape of K,V
        #unsqueeze(0) to add a batch dimension
        #src = src.unsqueeze(0)
        src = src.permute(1,0,2)
        _, bs, w = src.shape
        #src = src.flatten(2).permute(2, 0, 1)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # might not have to do this
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        #mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        if self.pre_norm:
            memory = self.pre_norm(src)
        else:
            memory = src

        hs = self.decoder(tgt, memory,
                          query_pos=query_embed,
                          num_frames=None, seq_len=w)
        return hs.transpose(1, 2)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ObjDecoder(nn.Module):
    def __init__(self, 
                 transformer, 
                 num_classes, 
                 num_queries, 
                 feature_dim=768, 
                 aux_loss=False,
                 pred_traj=True,
                 num_frames=4,
                 patches_per_frame=512,
                 backbone='LaviLa',
                 self_attn=False):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.hidden_dim = hidden_dim
        self.init_proj_layers()
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pred_traj = pred_traj
        if self.num_queries==1:
            self.n_decode = 10
            self.query_index = nn.Embedding(self.n_decode, hidden_dim)  # decode one query into many boxes in one frame 
        else:
            self.n_decode = 1

        self.aux_loss = aux_loss
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patches_per_frame + 1,
                        hidden_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        if self_attn:
            self.cls_embed = nn.Embedding(1, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
            self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=4)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temporal_embed, std=.02)
        self.proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.num_frames = num_frames
        self.init_obj_model()

    def init_proj_layers(self):
        # for NCE
        # self.txt_proj = nn.Sequential(nn.ReLU(), #  #: Uncomment this line to use the projection layer
        #     nn.Linear(self.feature_dim, 512))
        # self.vid_proj = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 512)) # : hidden v/s feature dim
        pass
        

    def init_obj_model(self):
        # self.obj_proj = nn.Sequential( # check this
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 512),
        # )
        self.obj_proj = nn.Parameter(torch.empty(self.hidden_dim,512))
        nn.init.normal_(self.obj_proj, std=self.hidden_dim ** -0.5)
        self.img_proj = nn.Parameter(torch.empty(self.hidden_dim,512))
        nn.init.normal_(self.img_proj, std=self.hidden_dim ** -0.5)

    def forward(self, features, use_checkpoint=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        
        Customized version:
            except: B,H,W,C
        """
        features = self.proj(features)
        #features = rearrange(features, 'b h w c -> b c h w')            #in our case, B,1,dim
        #B,_,T,W = features.shape                                    
        #mask = features.new_zeros(B,T,W).bool()
        #pos = self.construct_3d_pos_embed(T)                            #might not have to do this
        #pos = rearrange(pos, 'b h w c -> b c h w')

        hs = self.transformer(features, self.query_embed.weight)
        cond_hs = hs[:,:,:-1,:] 
        outputs_class = self.class_embed(cond_hs)   
        outputs_coord = self.bbox_embed(cond_hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        #hs = hs.permute(1,0,2,3)
        img_embed = hs[-1,:][:,-1,...] @ self.img_proj #: check this
        obj_embeds = hs[-1,:][:,:-1,...] @ self.obj_proj #: check this
        return out, img_embed, obj_embeds

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                num_frames: Optional[int] = 4,
                seq_len: Optional[int] = 196):
        output = tgt

        intermediate = []
        Q, B = output.shape[:2]
        # attn_rollout = torch.ones(B,qQ,memory.shape[0]).to(output.device)
        attns, self_attns= [],[]
        #print('0---',output.shape,'---')
        for layer_i,layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=query_pos, counter=layer_i,
                           num_frames = num_frames, seq_len=seq_len)
            # attns.append(attn.view(B,Q,4,16,16))
            # self_attns.append(self_attn[28][0])
            # attn_rollout  = attn_rollout*attn
            # plot_attn_map(attn.view(B,Q,4,16,16)[27][0].detach().cpu(),name=str(layer_i) )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # attn_rollout = attn_rollout.view(B,Q,4,16,16)
        # attns = torch.stack(attns).sum(0)
        # self_attns = torch.stack(self_attns)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sa_first=True):
        super().__init__()
        self.sa_first = sa_first

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                    counter: Optional[Tensor] = None,
                    num_frames: Optional[int] = 4,
                    seq_len: Optional[int] = 196):
        
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=memory,
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(0,1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    counter: Optional[Tensor] = None,
                    num_frames: Optional[int] = 4,
                    seq_len: Optional[int] = 196):
        if self.sa_first:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            #print('1---',tgt2.shape, memory.shape,'---')
            tgt2, self_attn = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            #print('2---',tgt2.shape, memory.shape,'---')
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                        key=self.with_pos_embed(memory, pos),
                        value=memory, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)

        else:
            tgt2 = self.norm1(tgt)
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, self_attn = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                counter: Optional[Tensor] = None,
                num_frames: Optional[int] = 4,
                seq_len: Optional[int] = 196):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, 
                                    query_pos, counter, num_frames, seq_len)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, 
                                 query_pos, counter, num_frames, seq_len)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



if __name__ == '__main__':
    pass