import torch
import torch.optim as optim
import models.lr_scheduler as lr_scheduler
import time
import os
import wandb

from models import model as model_hub
from models.dataloader import CUD_TrainLoader
from datetime import datetime
from models.utils import apply_stencil_mask
from torchvision.utils import save_image


class CUD:
    def __init__(self, args, now=None):

        self.start_time = time.time()
        self.args = args

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available of being set, not must.
        self.loader_train = self.__init_data_loader()

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            self.model_cud = self.__init_model()
            self.optimizer_cud = self.__init_optimizer()
            self.scheduler_cud = self.__set_scheduler()

            if self.args.mode == 'calibrate':
                self.model_cud.state_dict(torch.load(self.args.model_g_path))
                print('Model loaded successfully!!!')

        self.criterion = self.__init_criterion()

        if self.args.use_wandb:
            if self.args.mode == 'train':
                wandb.watch(self.model_cud)

        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.args.saved_model_directory + '_' + now_time

        if not os.path.exists(self.saved_model_directory):
            os.mkdir(self.saved_model_directory)

    def __train_cud(self, epoch):
        self.model_cud.train()

        for batch_idx, (data, target) in enumerate(self.loader_train):
            self.optimizer_cud.zero_grad()

            (x_img, x_feat), (target, target_feat) = data, target

            x_img = torch.cat((x_img, x_feat), dim=1)

            target_identity_input = torch.cat((target, target_feat), dim=1)

            x_img = x_img.to(self.device)
            target = target.to(self.device)

            output = self.model_cud(x_img)
            output = torch.clamp(output, 0.0, 1.0)

            output = apply_stencil_mask(_input=x_img, _output=output, _target=target)

            output_identity = self.model_cud(target_identity_input)
            output_identity = torch.clamp(output_identity, 0.0, 1.0)

            # the _input is original x_img as the target_identity_input makes stencil None
            output_identity = apply_stencil_mask(_input=x_img, _output=output_identity, _target=target)

            # compute the loss
            lab_loss, ssim_loss = self.criterion(x_img, output, target)

            # _input is converted to target in identity loss
            lab_loss_identity, ssim_loss_identity = self.criterion(target, output_identity, target, is_identity=True)

            # scale the loss
            # lab_loss = lab_loss * 3
            loss = lab_loss + lab_loss_identity + ssim_loss + ssim_loss_identity

            loss.backward()
            self.optimizer_cud.step()
            self.scheduler_cud.step()

            if batch_idx % self.args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] \t total_loss: {:.6f} \n'
                    'lab_loss: {:.6f} \t'
                    'ssim_loss: {:.6f} \t'
                    'lab_loss_identity: {:.6f} \t'
                    'ssim_loss_identity: {:.6f} \t'
                    .format(
                        epoch, batch_idx * self.args.batch_size, len(self.loader_train.dataset), 100. * batch_idx / len(self.loader_train),
                        loss.item() / self.args.batch_size,
                        lab_loss.item() / self.args.batch_size,
                        ssim_loss.item() / self.args.batch_size,
                        lab_loss_identity.item() / self.args.batch_size,
                        ssim_loss_identity.item() / self.args.batch_size,
                    ))

                if self.args.use_wandb:
                    wandb.log(
                        {'total_loss': loss.item() / self.args.batch_size,
                         'lab_loss': lab_loss.item() / self.args.batch_size,
                         'lab_identity_loss': lab_loss_identity.item() / self.args.batch_size,
                         'ssim_loss': ssim_loss.item() / self.args.batch_size,
                         'ssim_loss_identity': ssim_loss_identity.item() / self.args.batch_size,
                         })

            # end of epoch log
            # if batch_idx + 1 == self.loader_train.__len__() and self.args.use_wandb:
            #     wandb.log({'input': [wandb.Image(x.cpu().detach())]})
            #     wandb.log({'target': [wandb.Image(target.cpu().detach())]})
            #     wandb.log({'output': [wandb.Image(output.cpu().detach())]})

    def start_train(self):
        model_name = 'model'

        for epoch in range(1, self.args.epoch + 1):

            if self.args.mode == 'train' or self.args.mode == 'calibrate':
                model_name = 'CUD'
                self.__train_cud(epoch)

            print("{} epoch elapsed time : {}".format(epoch, time.time() - self.start_time))

            if epoch % self.args.save_interval == 0:

                self.save_model(model_name, epoch)

    def save_model(self, model_name, epoch, batch_idx=None):
        # normal location
        file_path = self.saved_model_directory + '/'
        if batch_idx is None:
            file_format_cud = file_path + model_name + str(epoch) + '.pt'
        else:
            file_format_cud = file_path + model_name + str(epoch) + '_' + str(batch_idx) + '.pt'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            torch.save(self.model_cud.state_dict(), file_format_cud)

        print("Model Saved!!")

    def __init_data_loader(self):
        # pin_memory = use CPU on data loader during GPU is training
        loader_cud = CUD_TrainLoader(dataset_path=self.args.train_x_path,
                                     label_path=self.args.train_y_path,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.worker,
                                     pin_memory=self.args.pin_memory)

        return loader_cud.Loader

    def __init_model(self):
        model_cud = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            model_cud = torch.nn.DataParallel(
                model_hub.CUD_NET(num_points=self.args.points).to(self.device))

        return model_cud

    def __init_criterion(self):
        cud_loss = model_hub.CUD_Loss().to(self.device)

        return cud_loss

    def __init_optimizer(self):
        optimizer_cud = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            if self.args.model == 'CUD':
                optimizer_cud = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_cud.parameters()),
                                                 lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
            else:
                optimizer_cud = optim.Adam(self.model_cud.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        return optimizer_cud

    def __set_scheduler(self):
        scheduler_cud = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            scheduler_cud = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer_cud,
                                                              warmup_steps=1,
                                                              t_total=self.loader_train.__len__(),
                                                              cycles=0.1,
                                                              last_epoch=-1)

        return scheduler_cud
