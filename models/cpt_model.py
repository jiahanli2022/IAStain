import numpy as np
import torch
import torch.nn as nn

from models.asp_loss import AdaptiveSupervisedPatchNCELoss
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util
import torch.nn.functional as F

from copy import deepcopy

class CPTModel(BaseModel):
    """ Contrastive Paired Translation (CPT).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.5, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--temperature', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        # FDL:
        parser.add_argument('--lambda_gp', type=float, default=1.0, help='weight for Gaussian Pyramid reconstruction loss')
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        parser.add_argument('--lambda_asp', type=float, default=0.0, help='weight for ASP loss')
        parser.add_argument('--asp_loss_mode', type=str, default='none', help='"scheduler_lookup" options for the ASP loss. Options for both are listed in Fig. 3 of the paper.')
        parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsample in G')
        parser.add_argument('--use_scl_finegrained', action='store_true')
        

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'P']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netP = nn.Sequential(nn.Linear(3+128+256*3, 256), nn.ReLU(), nn.Linear(256, 256)).to(self.device)


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_P)

            if self.opt.lambda_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = torch.nn.L1Loss().to(self.device)
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)
                self.loss_names += ['GP']

            if self.opt.lambda_asp > 0:
                self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['ASP']

            if self.opt.lambda_scl > 0:
                self.loss_names += ['SCL']


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0 or self.opt.lambda_asp > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_P.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_P.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.phase == 'train' and self.opt.use_mask:
            self.Mask = input['Mask'].to(self.device)
            if self.opt.use_dino:
                self.Freq = input['Freq'].to(self.device)
                if self.opt.use_scl:
                    self.SCL_Pool = input['SCL_Pool'].squeeze(0)
                    self.additional_SCL_NegPool = input['additional_SCL_NegPool'].squeeze(0)
                    self.Cluster_Number = input['Cluster_Number'].squeeze(0)

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']

    def forward(self):
        # self.netG.print()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if len(self.real_A.shape) != 4:
            assert len(self.real_A) == 3
            self.real_A, self.real_B = self.real_A.unsqueeze(0), self.real_B.unsqueeze(0)

            if self.opt.phase == 'train' and self.opt.use_mask:
                self.Mask  = self.Mask.unsqueeze(0)
                if self.opt.use_dino:
                    self.Freq = self.Freq.unsqueeze(0)            
        else:
            if self.opt.phase == 'train' and self.opt.use_mask and self.opt.use_scl:
                self.SCL_Pool = self.SCL_Pool.squeeze(0)
                self.additional_SCL_NegPool = self.additional_SCL_NegPool.squeeze(0)
                self.Cluster_Number = self.Cluster_Number.squeeze(0)

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        if self.opt.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        loss_NCE_all = self.loss_NCE

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE_Y = 0.0
        loss_NCE_all += self.loss_NCE_Y

        # FDL: NCE between the noisy pairs (fake_B and real_B)
        if self.opt.lambda_asp > 0:
            self.loss_ASP = self.calculate_NCE_loss(feat_real_B, feat_fake_B, self.netF, self.nce_layers, paired=True)
        else:
            self.loss_ASP = 0.0
        loss_NCE_all += self.loss_ASP

        # FDL: compute loss on Gaussian pyramids
        if self.opt.lambda_gp > 0:
            p_fake_B = self.P(self.fake_B)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_GP = torch.mean(torch.stack(loss_pyramid)) * self.opt.lambda_gp
        else:
            self.loss_GP = 0

        if self.opt.use_scl and self.opt.lambda_scl > 0:
            self.loss_SCL = self.calculate_SCL_loss(self.fake_B, self.real_B) * self.opt.lambda_scl
        else:
            self.loss_SCL = 0

        self.loss_G = self.loss_G_GAN + loss_NCE_all + self.loss_GP + self.loss_SCL
        return self.loss_G

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers, paired=False):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        if self.opt.phase == 'train' and self.opt.use_mask:
            if self.opt.use_dino:
                feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None, self.Mask, self.Freq)
            else:
                feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None, self.Mask, None)
        else:
            feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None, None, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            else:
                loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def info_nce_loss(self, anchor, pos_sample, additional_negatives, temperature=0.07):
        anchor = F.normalize(anchor, dim=1)       # (1, 256)
        positives = F.normalize(pos_sample, dim=1) # (K, 256)

        additional_negatives = F.normalize(additional_negatives, dim=1) # (L, 256)

        pos_sim = torch.exp(torch.matmul(anchor, positives.t()) / temperature)  # (1, K)
        additional_neg_sim = torch.exp(torch.matmul(anchor, additional_negatives.t()) / temperature)  # (1, L)
        
        loss = -torch.log(torch.sum(pos_sim) / (torch.sum(pos_sim) + torch.sum(additional_neg_sim)  + 1e-16))

        return loss
            
    def calculate_SCL_loss(self, fake_B_img, real_B_img):

        additional_neg_batch = self.additional_SCL_NegPool.to(self.device) # [8, 3, 256, 256]
        additional_neg_G_feat = torch.concat([F.adaptive_avg_pool2d(feat, output_size=(1, 1)).squeeze() for feat in self.netG(additional_neg_batch, self.nce_layers, encode_only=True)], 1)
        additional_neg_P_feat = self.netP(additional_neg_G_feat).detach()  # [8, 256]

        fake_B_batch = fake_B_img.unfold(2, 256, 256).unfold(3, 256, 256).permute(0, 2, 3, 1, 4, 5).reshape(-1, fake_B_img.size(1), 256, 256) #torch.Size([4, 3, 256, 256])
        anchor_G_feat = torch.concat([F.adaptive_avg_pool2d(feat, output_size=(1, 1)).squeeze() for feat in self.netG(fake_B_batch, self.nce_layers, encode_only=True)], 1)
        anchor_P_feat = self.netP(anchor_G_feat)

        real_B_batch = real_B_img.unfold(2, 256, 256).unfold(3, 256, 256).permute(0, 2, 3, 1, 4, 5).reshape(-1, real_B_img.size(1), 256, 256) #torch.Size([4, 3, 256, 256])
        real_B_G_feat = torch.concat([F.adaptive_avg_pool2d(feat, output_size=(1, 1)).squeeze() for feat in self.netG(real_B_batch, self.nce_layers, encode_only=True)], 1)
        real_B_P_feat = self.netP(real_B_G_feat).detach()

        loss = 0

        for i in range(len(self.Cluster_Number)):
            anchor_sample = anchor_P_feat[i:i+1]
            pos_sample = real_B_P_feat[i:i+1]

            loss += self.info_nce_loss(anchor_sample, pos_sample, additional_neg_P_feat, self.opt.temperature)
            
        return loss / len(self.Cluster_Number)