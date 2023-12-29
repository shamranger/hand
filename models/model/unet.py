from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer
from models.model.scene_model import create_scene_model
from models.base import MODEL

@MODEL.register()
class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()

        # self.d_x = cfg.d_x # 27
        self.d_x = 11700
        self.d_model = cfg.d_model # 512
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x

        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.condition_layers = nn.Sequential(
            nn.Conv1d(102, 512*16, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model,
                    self.transformer_num_heads,
                    self.transformer_dim_head,
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        x_t = x_t.reshape(x_t.shape[0], -1)
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1) # 32.1.11700
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)  #32,1024
        # print(t_emb.shape)  torch.Size([32, 512])

        t_emb = self.time_embed(t_emb)
        # print(t_emb.shape) torch.Size([32, 1024])


        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>  32, 512, 1

        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim> torch.Size([32, 512, 1]) torch.Size([32, 16, 512])

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            # print(B, DX, TX) 32, 11700, 1
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        condition_hand = cond['control'].unsqueeze(1)
        condition_hand = rearrange(condition_hand, 'b l c -> b c l')
        condition_hand = self.condition_layers(condition_hand)
        condition_hand = condition_hand.reshape(condition_hand.shape[0], 512, 16)
        condition_hand = rearrange(condition_hand, 'b l c -> b c l')


        for i in range(self.nblocks): #4
            h = self.layers[i * 3 + 0](h, t_emb)
            # print(1, h.shape) torch.Size([32, 512, 1])
            h = self.layers[i * 3 + 1](h, context=cond['obj_vertices'])
            # print(2, h.shape)
            h = self.layers[i * 3 + 2](h, context=condition_hand)
        # print(1,h.shape) torch.Size([32, 512, 1])
        h = self.out_layers(h)
        # print(2,h.shape) torch.Size([32, 27, 1])
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            condition = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            condition = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        # elif self.scene_model_name == 'PointNet2':
        #     b = data['contact_ref'].shape[0]
        #     obj_vertices = data['obj_vertices'].to(torch.float32)
        #     control = data['control'].to(torch.float32)
        #     _, scene_feat_list = self.scene_model(obj_vertices, control)
        #     condition = scene_feat_list[-1].transpose(1, 2)
        elif self.scene_model_name == 'PointNet2':
            b = data['contact_ref'].shape[0]
            obj_vertices = data['obj_vertices'].to(torch.float32)
            condition = {}
            condition['control'] = data['control'].to(torch.float32)
            # condition = self.scene_model(obj_vertices, control)
            _, scene_feat_list = self.scene_model(obj_vertices, condition['control'])
            condition['obj_vertices'] = scene_feat_list[-1].transpose(1, 2)
        else:
            raise Exception('Unexcepted scene model.')

        return condition
