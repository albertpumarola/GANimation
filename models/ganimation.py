import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np


class GANimation(BaseModel):
    def __init__(self, opt):
        super(GANimation, self).__init__(opt)
        self._name = 'GANimation'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G, device_ids=self._gpu_ids)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D, device_ids=self._gpu_ids)
        self._D.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name('generator_wasserstein_gan', c_dim=self._opt.cond_nc)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_wasserstein_gan', c_dim=self._opt.cond_nc)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_real_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_real_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        self._input_desired_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        self._input_real_img_path = None
        self._input_real_cond_path = None

    def _init_losses(self):
        # define loss functions
        self._criterion_cycle = torch.nn.L1Loss().cuda()
        self._criterion_D_cond = torch.nn.MSELoss().cuda()

        # init losses G
        self._loss_g_fake = Variable(self._Tensor([0]))
        self._loss_g_cond = Variable(self._Tensor([0]))
        self._loss_g_cyc = Variable(self._Tensor([0]))
        self._loss_g_mask_1 = Variable(self._Tensor([0]))
        self._loss_g_mask_2 = Variable(self._Tensor([0]))
        self._loss_g_idt = Variable(self._Tensor([0]))
        self._loss_g_masked_fake = Variable(self._Tensor([0]))
        self._loss_g_masked_cond = Variable(self._Tensor([0]))
        self._loss_g_mask_1_smooth = Variable(self._Tensor([0]))
        self._loss_g_mask_2_smooth = Variable(self._Tensor([0]))
        self._loss_rec_real_img_rgb = Variable(self._Tensor([0]))
        self._loss_g_fake_imgs_smooth = Variable(self._Tensor([0]))
        self._loss_g_unmasked_rgb = Variable(self._Tensor([0]))

        # init losses D
        self._loss_d_real = Variable(self._Tensor([0]))
        self._loss_d_cond = Variable(self._Tensor([0]))
        self._loss_d_fake = Variable(self._Tensor([0]))
        self._loss_d_gp = Variable(self._Tensor([0]))

    def set_input(self, input):
        self._input_real_img.resize_(input['real_img'].size()).copy_(input['real_img'])
        self._input_real_cond.resize_(input['real_cond'].size()).copy_(input['real_cond'])
        self._input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
        self._input_real_id = input['sample_id']
        self._input_real_img_path = input['real_img_path']

        if len(self._gpu_ids) > 0:
            self._input_real_img = self._input_real_img.cuda(self._gpu_ids[0], async=True)
            self._input_real_cond = self._input_real_cond.cuda(self._gpu_ids[0], async=True)
            self._input_desired_cond = self._input_desired_cond.cuda(self._gpu_ids[0], async=True)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    # get image paths
    def get_image_paths(self):
        return OrderedDict([('real_img', self._input_real_img_path)])

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        if not self._is_train:
            # convert tensor to variables
            real_img = Variable(self._input_real_img, volatile=True)
            real_cond = Variable(self._input_real_cond, volatile=True)
            desired_cond = Variable(self._input_desired_cond, volatile=True)

            # generate fake images
            fake_imgs, fake_img_mask = self._G.forward(real_img, desired_cond)
            fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
            fake_imgs_masked = fake_img_mask * real_img + (1 - fake_img_mask) * fake_imgs

            rec_real_img_rgb, rec_real_img_mask = self._G.forward(fake_imgs_masked, real_cond)
            rec_real_img_mask = self._do_if_necessary_saturate_mask(rec_real_img_mask, saturate=self._opt.do_saturate_mask)
            rec_real_imgs = rec_real_img_mask * fake_imgs_masked + (1 - rec_real_img_mask) * rec_real_img_rgb

            imgs = None
            data = None
            if return_estimates:
                # normalize mask for better visualization
                fake_img_mask_max = fake_imgs_masked.view(fake_img_mask.size(0), -1).max(-1)[0]
                fake_img_mask_max = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(fake_img_mask_max, -1), -1), -1)
                # fake_img_mask_norm = fake_img_mask / fake_img_mask_max
                fake_img_mask_norm = fake_img_mask

                # generate images
                im_real_img = util.tensor2im(real_img.data)
                im_fake_imgs = util.tensor2im(fake_imgs.data)
                im_fake_img_mask_norm = util.tensor2maskim(fake_img_mask_norm.data)
                im_fake_imgs_masked = util.tensor2im(fake_imgs_masked.data)
                im_rec_imgs = util.tensor2im(rec_real_img_rgb.data)
                im_rec_img_mask_norm = util.tensor2maskim(rec_real_img_mask.data)
                im_rec_imgs_masked = util.tensor2im(rec_real_imgs.data)
                im_concat_img = np.concatenate([im_real_img, im_fake_imgs_masked, im_fake_img_mask_norm, im_fake_imgs,
                                                im_rec_imgs, im_rec_img_mask_norm, im_rec_imgs_masked],
                                               1)

                im_real_img_batch = util.tensor2im(real_img.data, idx=-1, nrows=1)
                im_fake_imgs_batch = util.tensor2im(fake_imgs.data, idx=-1, nrows=1)
                im_fake_img_mask_norm_batch = util.tensor2maskim(fake_img_mask_norm.data, idx=-1, nrows=1)
                im_fake_imgs_masked_batch = util.tensor2im(fake_imgs_masked.data, idx=-1, nrows=1)
                im_concat_img_batch = np.concatenate([im_real_img_batch, im_fake_imgs_masked_batch,
                                                      im_fake_img_mask_norm_batch, im_fake_imgs_batch],
                                                     1)

                imgs = OrderedDict([('real_img', im_real_img),
                                    ('fake_imgs', im_fake_imgs),
                                    ('fake_img_mask', im_fake_img_mask_norm),
                                    ('fake_imgs_masked', im_fake_imgs_masked),
                                    ('concat', im_concat_img),
                                    ('real_img_batch', im_real_img_batch),
                                    ('fake_imgs_batch', im_fake_imgs_batch),
                                    ('fake_img_mask_batch', im_fake_img_mask_norm_batch),
                                    ('fake_imgs_masked_batch', im_fake_imgs_masked_batch),
                                    ('concat_batch', im_concat_img_batch),
                                    ])

                data = OrderedDict([('real_path', self._input_real_img_path),
                                    ('desired_cond', desired_cond.data[0, ...].cpu().numpy().astype('str'))
                                    ])

            # keep data for visualization
            if keep_data_for_visuals:
                self._vis_real_img = util.tensor2im(self._input_real_img)
                self._vis_fake_img_unmasked = util.tensor2im(fake_imgs.data)
                self._vis_fake_img = util.tensor2im(fake_imgs_masked.data)
                self._vis_fake_img_mask = util.tensor2maskim(fake_img_mask.data)
                self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
                self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()
                self._vis_batch_real_img = util.tensor2im(self._input_real_img, idx=-1)
                self._vis_batch_fake_img_mask = util.tensor2maskim(fake_img_mask.data, idx=-1)
                self._vis_batch_fake_img = util.tensor2im(fake_imgs_masked.data, idx=-1)

            return imgs, data

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_real_img.size(0)
            self._real_img = Variable(self._input_real_img)
            self._real_cond = Variable(self._input_real_cond)
            self._desired_cond = Variable(self._input_desired_cond)

            # train D
            loss_D, fake_imgs_masked = self._forward_D()
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

            loss_D_gp= self._gradinet_penalty_D(fake_imgs_masked)
            self._optimizer_D.zero_grad()
            loss_D_gp.backward()
            self._optimizer_D.step()

            # train G
            if train_generator:
                loss_G = self._forward_G(keep_data_for_visuals)
                self._optimizer_G.zero_grad()
                loss_G.backward()
                self._optimizer_G.step()

    def _forward_G(self, keep_data_for_visuals):
        # generate fake images
        fake_imgs, fake_img_mask = self._G.forward(self._real_img, self._desired_cond)
        fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
        fake_imgs_masked = fake_img_mask * self._real_img + (1 - fake_img_mask) * fake_imgs

        # D(G(Ic1, c2)*M) masked
        d_fake_desired_img_masked_prob, d_fake_desired_img_masked_cond = self._D.forward(fake_imgs_masked)
        self._loss_g_masked_fake = self._compute_loss_D(d_fake_desired_img_masked_prob, True) * self._opt.lambda_D_prob
        self._loss_g_masked_cond = self._criterion_D_cond(d_fake_desired_img_masked_cond, self._desired_cond) / self._B * self._opt.lambda_D_cond

        # G(G(Ic1,c2), c1)
        rec_real_img_rgb, rec_real_img_mask = self._G.forward(fake_imgs_masked, self._real_cond)
        rec_real_img_mask = self._do_if_necessary_saturate_mask(rec_real_img_mask, saturate=self._opt.do_saturate_mask)
        rec_real_imgs = rec_real_img_mask * fake_imgs_masked + (1 - rec_real_img_mask) * rec_real_img_rgb

        # l_cyc(G(G(Ic1,c2), c1)*M)
        self._loss_g_cyc = self._criterion_cycle(rec_real_imgs, self._real_img) * self._opt.lambda_cyc

        # loss mask
        self._loss_g_mask_1 = torch.mean(fake_img_mask) * self._opt.lambda_mask
        self._loss_g_mask_2 = torch.mean(rec_real_img_mask) * self._opt.lambda_mask
        self._loss_g_mask_1_smooth = self._compute_loss_smooth(fake_img_mask) * self._opt.lambda_mask_smooth
        self._loss_g_mask_2_smooth = self._compute_loss_smooth(rec_real_img_mask) * self._opt.lambda_mask_smooth

        # keep data for visualization
        if keep_data_for_visuals:
            self._vis_real_img = util.tensor2im(self._input_real_img)
            self._vis_fake_img_unmasked = util.tensor2im(fake_imgs.data)
            self._vis_fake_img = util.tensor2im(fake_imgs_masked.data)
            self._vis_fake_img_mask = util.tensor2maskim(fake_img_mask.data)
            self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
            self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()
            self._vis_batch_real_img = util.tensor2im(self._input_real_img, idx=-1)
            self._vis_batch_fake_img_mask = util.tensor2maskim(fake_img_mask.data, idx=-1)
            self._vis_batch_fake_img = util.tensor2im(fake_imgs_masked.data, idx=-1)
            self._vis_rec_img_unmasked = util.tensor2im(rec_real_img_rgb.data)
            self._vis_rec_real_img = util.tensor2im(rec_real_imgs.data)
            self._vis_rec_real_img_mask = util.tensor2maskim(rec_real_img_mask.data)
            self._vis_batch_rec_real_img = util.tensor2im(rec_real_imgs.data, idx=-1)

        # combine losses
        return self._loss_g_masked_fake + self._loss_g_masked_cond + \
               self._loss_g_cyc + \
               self._loss_g_mask_1 + self._loss_g_mask_2 + \
               self._loss_g_mask_1_smooth + self._loss_g_mask_2_smooth

    def _forward_D(self):
        # generate fake images
        fake_imgs, fake_img_mask = self._G.forward(self._real_img, self._desired_cond)
        fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
        fake_imgs_masked = fake_img_mask * self._real_img + (1 - fake_img_mask) * fake_imgs

        # D(real_I)
        d_real_img_prob, d_real_img_cond = self._D.forward(self._real_img)
        self._loss_d_real = self._compute_loss_D(d_real_img_prob, True) * self._opt.lambda_D_prob
        self._loss_d_cond = self._criterion_D_cond(d_real_img_cond, self._real_cond) / self._B * self._opt.lambda_D_cond

        # D(fake_I)
        d_fake_desired_img_prob, _ = self._D.forward(fake_imgs_masked.detach())
        self._loss_d_fake = self._compute_loss_D(d_fake_desired_img_prob, False) * self._opt.lambda_D_prob

        # combine losses
        return self._loss_d_real + self._loss_d_cond + self._loss_d_fake, fake_imgs_masked

    def _gradinet_penalty_D(self, fake_imgs_masked):
        # interpolate sample
        alpha = torch.rand(self._B, 1, 1, 1).cuda().expand_as(self._real_img)
        interpolated = Variable(alpha * self._real_img.data + (1 - alpha) * fake_imgs_masked.data, requires_grad=True)
        interpolated_prob, _ = self._D(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._opt.lambda_D_gp

        return self._loss_d_gp

    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_fake', self._loss_g_fake.data[0]),
                                 ('g_cond', self._loss_g_cond.data[0]),
                                 ('g_mskd_fake', self._loss_g_masked_fake.data[0]),
                                 ('g_mskd_cond', self._loss_g_masked_cond.data[0]),
                                 ('g_cyc', self._loss_g_cyc.data[0]),
                                 ('g_rgb', self._loss_rec_real_img_rgb.data[0]),
                                 ('g_rgb_un', self._loss_g_unmasked_rgb.data[0]),
                                 ('g_rgb_s', self._loss_g_fake_imgs_smooth.data[0]),
                                 ('g_m1', self._loss_g_mask_1.data[0]),
                                 ('g_m2', self._loss_g_mask_2.data[0]),
                                 ('g_m1_s', self._loss_g_mask_1_smooth.data[0]),
                                 ('g_m2_s', self._loss_g_mask_2_smooth.data[0]),
                                 ('g_idt', self._loss_g_idt.data[0]),
                                 ('d_real', self._loss_d_real.data[0]),
                                 ('d_cond', self._loss_d_cond.data[0]),
                                 ('d_fake', self._loss_d_fake.data[0]),
                                 ('d_gp', self._loss_d_gp.data[0])])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # input visuals
        title_input_img = os.path.basename(self._input_real_img_path[0])
        visuals['1_input_img'] = plot_utils.plot_au(self._vis_real_img, self._vis_real_cond, title=title_input_img)
        visuals['2_fake_img'] = plot_utils.plot_au(self._vis_fake_img, self._vis_desired_cond)
        visuals['3_rec_real_img'] = plot_utils.plot_au(self._vis_rec_real_img, self._vis_real_cond)
        visuals['4_fake_img_unmasked'] = self._vis_fake_img_unmasked
        visuals['5_fake_img_mask'] = self._vis_fake_img_mask
        visuals['6_rec_real_img_mask'] = self._vis_rec_real_img_mask
        visuals['7_cyc_img_unmasked'] = self._vis_fake_img_unmasked
        # visuals['8_fake_img_mask_sat'] = self._vis_fake_img_mask_saturated
        # visuals['9_rec_real_img_mask_sat'] = self._vis_rec_real_img_mask_saturated
        visuals['10_batch_real_img'] = self._vis_batch_real_img
        visuals['11_batch_fake_img'] = self._vis_batch_fake_img
        visuals['12_batch_fake_img_mask'] = self._vis_batch_fake_img_mask
        # visuals['11_idt_img'] = self._vis_idt_img

        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)

            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))

        # update learning rate D
        lr_decay_D = self._opt.lr_D / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' %  (self._current_lr_D + lr_decay_D, self._current_lr_D))

    def _l1_loss_with_target_gradients(self, input, target):
        return torch.sum(torch.abs(input - target)) / input.data.nelement()

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
