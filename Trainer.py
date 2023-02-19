import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from math import cos, pi
from denoising_diffusion import DenoisingDiffusion
from img_utils import SkinLoader
from ViT import ViT
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.backends.cudnn.benchmark = True

class CosineAnnealingWithWarmup:
    def __init__(self, optim, max_lr, warmup_steps, end_step, min_lr=0):
        self.optim = optim
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.end_step = end_step
        self.curr_iter = 0

        self.warmup_increase_per_step = (max_lr - min_lr) / warmup_steps
        self.decay_steps = end_step - warmup_steps

    def step(self):
        self.curr_iter += 1
        if self.curr_iter <= self.warmup_steps:
            lr = self.warmup_increase_per_step * self.curr_iter + self.min_lr
        elif self.warmup_steps < self.curr_iter <= self.end_step:
            lr = self.max_lr * 0.5 * (1 + cos(pi * (self.curr_iter - self.warmup_steps) / self.decay_steps))
        else:
            raise Exception('end of training')

        for param in self.optim.param_groups:
            param['lr'] = lr


class Trainer:
    def __init__(self, **kwargs):

        self.checkpoint_every = 10000
        self.progress_bar = None
        self.data_dir = kwargs['data_dir']

        if kwargs['from_checkpoint'] is not None:
            self.diffusion, self.batch_size, self.scheduler = torch.load(
                kwargs['from_checkpoint']).values()
        else:
            self.diffusion = DenoisingDiffusion(
                ViT(n_dim=kwargs['base_channels'], diffusion_timesteps=kwargs['diffusion_timesteps']),
                n_steps=kwargs['diffusion_timesteps']
            )
            self.batch_size = kwargs['batch_size']
            self.scheduler = CosineAnnealingWithWarmup(optim=Adam(self.diffusion.model.parameters()),
                                                       max_lr=kwargs['max_lr'],
                                                       warmup_steps=10000,
                                                       end_step=kwargs['iters'])
        self.diffusion = self.diffusion.to(device)
        # self.scheduler = CosineAnnealingWithWarmup(optim=Adam(self.diffusion.model.parameters()),
        #                                            max_lr=kwargs['max_lr'],
        #                                            warmup_steps=1000,
        #                                            end_step=kwargs['iters'])

    def checkpoint(self, path):
        PATH = f'{path}//ViTt retrain at {self.scheduler.curr_iter}.pt'
        self.scheduler.optim.zero_grad(set_to_none=True)
        torch.save({
            'diffusion': self.diffusion,
            'batch-size': self.batch_size,
            'scheduler': self.scheduler,
            # 'optimizer': self.scheduler.optim.state_dict(),
            # 'progress_bar': self.progress_bar,
        }, PATH)

    def train(self):
        temp = []
        avg = 0
        print(self.diffusion.model.num_params)
        scaler = torch.cuda.amp.GradScaler()
        try:
            while True:
                dataloader = DataLoader(SkinLoader(self.data_dir), batch_size=self.batch_size,
                                        shuffle=True, pin_memory=True, num_workers=4)
                with tqdm(dataloader, unit='Iterations', total=self.scheduler.end_step,
                          initial=self.scheduler.curr_iter) as progress_bar:
                    for batch in progress_bar:
                        progress_bar.set_description(f'Iteration {self.scheduler.curr_iter}')
                        self.scheduler.optim.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast():
                            loss = self.diffusion.simple_loss(batch.to(device))
                        scaler.scale(loss).backward()
                        self.scheduler.step()
                        scaler.step(self.scheduler.optim)
                        scaler.update()
                        progress_bar.set_postfix(loss=loss.item(), lr=self.scheduler.optim.param_groups[0]['lr'], prev_avg_loss = avg)
                        temp.append(loss.item())
                        if self.scheduler.curr_iter % 500 == 0:
                            avg = sum(temp) / len(temp)
                            temp = []
                        if self.scheduler.curr_iter % self.checkpoint_every == 0:
                            if torch.isnan(loss):
                                raise Exception('NaN loss')
                            self.checkpoint('checkpoints')
                    #     with torch.no_grad():
                    #         loss = self.diffusion.simple_loss(batch.to(device))
                    # temp.append(loss.item())
                print(f'Average loss at {self.scheduler.curr_iter}: {sum(temp) / len(temp)}')
        except (KeyboardInterrupt, Exception) as e:
            # self.checkpoint('checkpoints')
            raise e


if __name__ == '__main__':
    j = Trainer(
        from_checkpoint=None,
        data_dir="C:\\Users\\ricky\\Downloads\\Skins\\Skins",
        base_channels=2048,
        diffusion_timesteps=1000,
        batch_size=40,
        max_lr=1e-5,
        iters=1000000
    )
    torch.cuda.empty_cache()
    j.train()
    # "checkpoints/ViT at 55000.pt"  good skin model