from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.base import DIFFUSER
from models.dm.schedule import make_schedule_ddpm
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner

@DIFFUSER.register()
class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(DDPM, self).__init__()

        shape_feature_dim = cfg.shape_feature_dim
        latent_dim = cfg.latent_dim
        temporal_dim = cfg.temporal_dim
        in_seq_dim = cfg.in_seq_dim
        out_seq_dim = cfg.out_seq_dim
        control_dim = cfg.control_dim
        condition_dim = cfg.condition_dim
        seq_len = cfg.seq_len
        ref_dim = cfg.ref_dim
        n_stages = cfg.n_stages
        n_parts = cfg.n_parts
        input_obj_dim = cfg.input_obj_dim
        self.n_stages = cfg.n_stages
        self.n_parts = cfg.n_parts
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.out_seq_dim = out_seq_dim

        self.eps_model = eps_model
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type

        self.has_observation = has_obser # used in some task giving observation

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        
        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
                
        self.optimizer = None
        self.planner = None

    @property
    def device(self):
        return self.betas.device
    
    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Apply observation to x_t, if self.has_observation if False, this method will return the input

        Args:
            x_t: noisy x in step t
            data: original data provided by dataloader
        """
        ## has start observation, used in path planning and start-conditioned motion generation
        if self.has_observation and 'start' in data:
            start = data['start'] # <B, T, D>
            T = start.shape[1]
            x_t[:, 0:T, :] = start[:, 0:T, :].clone()
        
            if 'obser' in data:
                obser = data['obser']
                O = obser.shape[1]
                x_t[:, T:T+O, :] = obser.clone()
        
        return x_t
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['contact_ref'].shape[0] #32
        obj_vertices = data['obj_vertices']
        contact_ref = data['contact_ref']
        in_seq_1 = data['in_seq_1']
        in_seq_2 = data['in_seq_2']
        in_contact_flag_seq = data['in_contact_flag_seq']
        in_near_flag_seq = data['in_near_flag_seq']
        t_seq = data['t_seq']
        control = data['control']

        # print(B)
        # data = {
        #     'obj_vertices': obj_vertices,  # 32 2,1000,3
        #     'contact_ref': contact_ref,  #   3 2 5 7
        #     'in_seq_1': in_seq_1,  #   3 2 10 5 15
        #     'in_seq_2': in_seq_2,  #   3 2 10 5 15
        #     'in_contact_flag_seq': in_contact_flag_seq,  # 3 2 10 5
        #     'in_near_flag_seq': in_near_flag_seq,  # 3 2 10 5
        #     'out_seq_1': out_seq_1,  # 3 2 10 5 15
        #     'out_seq_2': out_seq_2,  # 3 2 10 5 15
        #     'out_contact_flag_seq': out_contact_flag_seq,  # 3 2 10 5
        #     'out_near_flag_seq': out_near_flag_seq,  # 3 2 10 5
        #     'stage_length': stage_length,  # 80 80 40
        #     't_seq': t_seq,  # 3 2 10
        #     'control': control,  # 102
        #     'pts_range': pts_range,  # 2 3 2
        #     'mano_beta': self.meta[seq_path]['data']['beta_mano'],  # 10
        #     'seq_path': seq_path
        # }

        in_seq = torch.cat([
            in_seq_1, in_seq_2, in_contact_flag_seq.unsqueeze(-1), in_near_flag_seq.unsqueeze(-1)
        ], dim=-1)
        contact_ref = contact_ref.unsqueeze(3).tile(1, 1, 1, self.seq_len, 1, 1)
        seq = torch.cat([contact_ref, in_seq], dim=-1)
        seq = seq.reshape(B, -1)
        ## randomly sample timesteps
        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            # print(ts)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
                # print(ts)
        else:
            raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(seq.view(B, -1), device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=seq, t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data)  #32,27

        ## predict noise
        condtion = self.eps_model.condition(data)  # 32,16,512
        output = self.eps_model(x_t, ts, condtion)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        output = self.apply_observation(output, data)  #32,27
        output = output.reshape(B, 3, 2, 10, 5, 39)
        state_out = output[:, :, :, :, :, 0:7]
        state_out = torch.mean(state_out, dim=3)

        out_seq_1 = output[:, :, :, :, :, 7:22]
        out_seq_2 = output[:, :, :, :, :, 22:37]
        out_contact_flag = torch.sigmoid(output[:, :, :, :, :, 37])
        out_near_flag = torch.sigmoid(output[:, :, :, :, :, 38])
        y = (torch.sigmoid(state_out[:, :, :, :, 0]), state_out[:, :, :, :, 1:], \
                out_seq_1, out_seq_2, out_contact_flag, out_near_flag)

        dir_weight, joints_weight = 100, 1
        self.cvae_all_loss = CVAEALLLoss(dir_weight, joints_weight)
        self.binary_loss, self.pos_L2_loss, \
        self.dir_L2_loss, self.KLD_loss, \
        self.seq_tip_loss, self.seq_joints_loss = self.cvae_all_loss(y)

        self.CAMSloss = self.binary_loss + self.pos_L2_loss + self.dir_L2_loss + \
                    self.KLD_loss + self.seq_tip_loss + self.seq_joints_loss

        ## calculate loss
        loss = self.criterion(output, noise) + self.CAMSloss

        return {'loss': loss}
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        pred_noise = self.eps_model(x_t, t, cond)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        ## sampling with mean updated by optimizer and planner
        if self.optimizer is not None:
            ## openai guided diffusion uses the input x to compute gradient, see
            ## https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L436
            ## But the original formular uses the computed mean?
            gradient = self.optimizer.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient
        if self.planner is not None:
            gradient = self.planner.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient

        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        x_t = self.apply_observation(x_t, data)
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            ## apply observation to x_t
            x_t = self.apply_observation(x_t, data)
            
            all_x_t.append(x_t)
        return torch.stack(all_x_t, dim=1)
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data))
        
        ksamples = torch.stack(ksamples, dim=1)
        
        ## for sequence, normalize and convert repr
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                ## the start observation frames are replace during sampling
                _, O, _ = data['start'].shape
            ksamples[..., O:, :] = data['normalizer'].unnormalize(ksamples[..., O:, :])
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                O = 1
                if self.has_observation and 'start' in data:
                    _, O, _ = data['start'].shape
                ksamples[..., O-1:, :] = torch.cumsum(ksamples[..., O-1:, :], dim=-2)
            else:
                raise Exception('Unsupported repr type.')
        
        return ksamples
    
    def set_optimizer(self, optimizer: Optimizer):
        """ Set optimizer for diffuser, the optimizer is used in sampling

        Args:
            optimizer: a Optimizer object that has a gradient method
        """
        self.optimizer = optimizer
    
    def set_planner(self, planner: Planner):
        """ Set planner for diffuser, the planner is used in sampling

        Args:
            planner: a Planner object that has a gradient method
        """
        self.planner = planner

class CVAEALLLoss(nn.Module):
    def __init__(self, lambda_dir=100, lambda_seq_joints=1) -> None:
        super().__init__()
        self.lambda_binary = 0.1
        self.lambda_pos = 500
        self.lambda_dir = lambda_dir
        # self.lambda_dir = 1
        self.lambda_KLD = 5

        self.lambda_seq_tip = 100
        self.lambda_seq_joints = lambda_seq_joints

    def forward(self, batch):

        gt_ref = batch['contact_ref']
        gt_seq_1 = batch['out_seq_1']
        gt_seq_2 = batch['out_seq_2']
        gt_c = batch['out_contact_flag_seq']
        gt_n = batch['out_near_flag_seq']
        pred_ref_flag = batch['pred_ref_flag']
        pred_ref = batch['pred_ref']
        pred_seq_1 = batch['pred_seq_1']
        pred_seq_2 = batch['pred_seq_2']
        pred_c = batch['pred_c_flag']
        pred_n = batch['pred_n_flag']
        mu = batch['mu']
        logvar = batch['logvar']

        gt_flag = gt_ref[:, :, :, :, 0]
        gt_ref = gt_ref[:, :, :, :, 1:]

        binary_loss = (-gt_flag*(pred_ref_flag+1e-6).log()-(1-gt_flag)*(1-pred_ref_flag+1e-6).log()).mean() \
                    + (-gt_c*(pred_c+1e-6).log()-(1-gt_c)*(1-pred_c+1e-6).log()).mean() \
                    + (-gt_n*(pred_n+1e-6).log()-(1-gt_n)*(1-pred_n+1e-6).log()).mean()

        gt_flag = gt_flag.unsqueeze(-1)
        pos_L2_loss = (gt_flag*(gt_ref[:, :, :, :, :3]-pred_ref[:, :, :, :, :3])**2).sum()/gt_flag.sum()
        dir_L2_loss = (gt_flag*(gt_ref[:, :, :, :, 3:]-pred_ref[:, :, :, :, 3:])**2).sum()/gt_flag.sum()
        KLD_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()

        gt_flag_1 = gt_flag[:, :-1].unsqueeze(-3)
        gt_flag_1 = torch.cat([torch.zeros_like(gt_flag_1[:, :1]), gt_flag_1], dim=1)
        gt_flag_2 = gt_flag.unsqueeze(-3)
        n_mask = gt_n.unsqueeze(-1)
        delta_seq_1 = (gt_flag_1 * n_mask * (gt_seq_1-pred_seq_1)**2).view(-1, 5, 3)
        delta_seq_2 = (gt_flag_2 * n_mask * (gt_seq_2-pred_seq_2)**2).view(-1, 5, 3)
        tip_loss = delta_seq_1[:, 4].mean() + delta_seq_2[:, 4].mean()
        joints_loss = delta_seq_1[:, :4].mean() + delta_seq_2[:, :4].mean()

        binary_loss = self.lambda_binary * binary_loss
        pos_L2_loss = self.lambda_pos * pos_L2_loss
        dir_L2_loss = self.lambda_dir * dir_L2_loss
        KLD_loss = self.lambda_KLD * KLD_loss
        tip_loss = self.lambda_seq_tip * tip_loss
        joints_loss = self.lambda_seq_joints * joints_loss

        return binary_loss, pos_L2_loss, dir_L2_loss, KLD_loss, tip_loss, joints_loss
