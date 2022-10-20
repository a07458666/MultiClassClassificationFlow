from audioop import reverse
import os
from math import log, pi

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils import data
from scipy.stats import norm

from module.flow import build_model


class PModel:

    @staticmethod
    def logprob(z):
        dim = z.size(-1)
        log_z = -0.5 * dim * log(2 * pi)
        return log_z - z.pow(2) / 2

    @staticmethod
    def prob(z):
        return 1/(2*pi)**0.5 * np.exp(-((z*z)/2))

    @staticmethod
    def invcdf(q):
        return norm.ppf(q)

    @staticmethod
    def sample(shape) -> np.ndarray:
        return np.random.normal(0, 1, shape)


class FlowSampler:

    def __init__(self, shape, flow_modules, num_blocks, gpu=0, pmodel=PModel) -> None:
        self.pmodel = pmodel
        self.shape = shape
        self.gpu = gpu

        input_dim = 1
        for dim in self.shape:
            input_dim *= dim
        self.input_dim = input_dim

        def cnf(input_dim, dims, num_blocks):
            dims = tuple(map(int, dims.split("-")))
            model = build_model(input_dim, dims, 1, num_blocks, True).cuda()
            return model
        
        self.prior = cnf(input_dim, flow_modules, num_blocks).cuda(self.gpu)


    def fit(self, x, epoch=10, lr=1e-2, save_model=False, save_dir=None, batch=32) -> list:
        self.prior.train()
        
        class MyDataset(data.Dataset):
            def __init__(self, x, transform=None):
                self.x = x
                self.transform = transform

            def __getitem__(self, index):
                x = self.x[index]

                if self.transform is not None:
                    x = self.transform(x)

                return x

            def __len__(self):
                return len(self.x)

        my_dataset = MyDataset(x=torch.Tensor(x).cuda(self.gpu))
        train_loader = data.DataLoader(my_dataset, shuffle=True, batch_size=batch)
        optimizer = optim.Adam(self.prior.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

        loss_list = []

        for i in tqdm(range(epoch)):
            for x in train_loader:
                loss =  - self.__logp(x).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
            scheduler.step()

            if save_model and save_dir is not None:
                path = os.path.join(save_dir, 'sampler_' + str(i).zfill(2) + '.pt')
                self.save(path)
        
        if save_model and save_dir is not None:
            path = os.path.join(save_dir, 'sampler_last.pt')
            self.save(path)
        
        return loss_list


    def save(self, path) -> None:
        torch.save(
            self.prior.state_dict(),
            path
        )


    def load(self, path) -> None:
        self.prior.load_state_dict(torch.load(path))


    def sample(self, n=1) -> torch.Tensor:
        self.prior.eval()

        with torch.no_grad():
            z = self.pmodel.sample((n, self.input_dim))
            z = torch.tensor(z).float().to(self.gpu)
            x = self.prior(z, torch.zeros(n, 1, 1).to(z), reverse=True)

            return x.view((-1,)+self.shape)


    def logprob(self, x) -> torch.Tensor:
        self.prior.eval()

        with torch.no_grad():
            return self.__logp(x)


    def __logp(self, x) -> torch.Tensor:
        x = x.view(x.size()[0], 1, -1)

        # delta_p = torch.zeros(x.size()).to(x)
        context = torch.zeros(x.size()[0], 1, 1).to(x)
        delta_p = torch.zeros(x.shape[0], x.shape[1], 1).to(x)
        # print("x : ", x.size())
        # print("context : ", context.size())
        # print("delta_p : ", delta_p.size())

        z, delta_log_p = self.prior(x, context, delta_p)
        
        log_p_z = self.pmodel.logprob(z).view(x.shape[0], -1).sum(1, keepdim=True)
        delta_log_p = delta_log_p.view(x.shape[0], 1, -1).sum(1)
        log_p_x = (log_p_z - delta_log_p)
        return log_p_x
