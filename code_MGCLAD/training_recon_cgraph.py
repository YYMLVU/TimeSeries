import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_recon": [],
            "train_cl": [],
            "val_total": [],
            "val_recon": [],
            "val_cl": []
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()
            model_mem = torch.cuda.memory_allocated() / (1024**2)
            print('model_mem:', model_mem)

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None, train_aug_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        # init_train_loss = self.evaluate(train_loader)
        # print(f"Init total train loss: {init_train_loss[1]:5f}")

        # if val_loader is not None:
        #     init_val_loss = self.evaluate(val_loader)
        #     print(f"Init total val loss: {init_val_loss[1]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            recon_b_losses = []
            cl_b_losses = []

            # Train on original data and augmented data
            for (xy , xy_aug) in zip(train_loader, train_aug_loader):
                x, y = xy
                x = x.to(self.device)
                y = y.to(self.device)
                x_aug, _ = xy_aug
                x_aug = x_aug.to(self.device)
                self.optimizer.zero_grad()

                recons, cl = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]

                loss1 = torch.sqrt(self.recon_criterion(x, recons))
                loss = loss1 + 0.1*cl
                loss.backward()
                self.optimizer.step()

                recon_b_losses.append(loss1.item())
                cl_b_losses.append(cl.item())

            recon_b_losses = np.array(recon_b_losses)
            cl_b_losses = np.array(cl_b_losses)

            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())
            cl_epoch_loss = np.sqrt((cl_b_losses ** 2).mean())

            total_epoch_loss = recon_epoch_loss + cl_epoch_loss

            self.losses["train_cl"].append(cl_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            recon_val_loss, cl_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                recon_val_loss, cl_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_cl"].append(cl_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                # if total_val_loss <= self.losses["val_total"][-1]:
                #     self.save(f"model.pt")
                if total_val_loss <= min(self.losses["val_total"]):
                    self.save(f"model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            max_mem = torch.cuda.max_memory_allocated() / (1024**2)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"recon_loss={recon_epoch_loss:.4f} "
                    f"cl_loss={cl_epoch_loss:.4f} "
                    f"total={total_epoch_loss:.4f}"
                )

                if val_loader is not None:
                    s += (
                        f" | v_recon_loss={recon_val_loss:.4f} "
                        f"v_cl_loss={cl_val_loss:.4f} "
                        f"v_total={total_val_loss:.4f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)
                print('max_mem:', max_mem)

        if val_loader is None:
            self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        cl_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)

                recons, cl = self.model(x, training=True)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]

                recon_loss = torch.sqrt(self.recon_criterion(x, recons))

                recon_losses.append(recon_loss.item())
                cl_losses.append(cl.item())

        # recon_losses = np.array(recon_losses)
        # recon_loss = np.sqrt((recon_losses ** 2).mean())

        recon_losses = np.array(recon_losses)
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        cl_losses = np.array(cl_losses)
        cl_loss = np.sqrt((cl_losses ** 2).mean())

        total_loss = recon_loss + cl_loss

        return recon_loss, cl_loss, total_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
