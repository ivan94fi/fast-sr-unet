from comet_ml import Experiment  # must be imported before torch

# isort: split

import os, utils
from utils import make_grid, plot_grads
import time
from pprint import pformat

args = utils.ARArgs()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

import data_loader as dl
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_ssim  # courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
import tqdm
import lpips  # courtesy of https://github.com/richzhang/PerceptualSimilarity
from models import Discriminator, \
    SRResNet  # courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
from pytorch_unet import SRUnet, UNet, SimpleResNet

import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module=r"torch.nn.functional",
)


if __name__ == '__main__':
    args = utils.ARArgs()

    experiment = Experiment(
        project_name="memorestore",
        auto_metric_logging=False,
        log_graph=False,
        disabled=args.no_comet,
    )

    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR
    n_epochs = args.N_EPOCHS

    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    if args.MODEL_NAME is not None:
        print("Loading model: ", args.MODEL_NAME)
        state_dict = torch.load(args.MODEL_NAME)
        model.load_state_dict(state_dict)

    critic = Discriminator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    critic_opt = torch.optim.Adam(lr=1e-4, params=critic.parameters())
    gan_opt = torch.optim.Adam(lr=1e-4, params=model.parameters())

    lpips_loss = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex = lpips.LPIPS(net='alex', version='0.1')
    ssim = pytorch_ssim.SSIM()

    model.to(device)
    lpips_loss.to(device)
    lpips_alex.to(device)
    critic.to(device)
    sigmoid = nn.Sigmoid()

    model_graph = {
        "generator": model,
        "discriminator": critic,
    }
    experiment.set_model_graph(pformat(model_graph))

    dataset_train = dl.ARDataLoader2(path=str(args.DATASET_DIR), crf=args.CRF, patch_size=96, eval=False, use_ar=True)
    dataset_test = dl.ARDataLoader2(path=str(args.DATASET_DIR), crf=args.CRF, patch_size=96, eval=True, use_ar=True)

    data_loader = DataLoader(dataset=dataset_train, batch_size=32, num_workers=12, shuffle=True,
                             pin_memory=True)
    data_loader_eval = DataLoader(dataset=dataset_test, batch_size=32, num_workers=12, shuffle=True,
                                  pin_memory=True)

    loss_discriminator = nn.BCEWithLogitsLoss()

    print(f"Total epochs: {n_epochs}; Steps per epoch: {len(data_loader)}")

    # setting loss weights
    w0, w1, l0 = args.W0, args.W1, args.L0

    for e in range(n_epochs):
        print("Epoch:", e)

        tqdm_ = tqdm.tqdm(data_loader, dynamic_ncols=True)
        step = 0
        for batch in tqdm_:
            model.train()
            critic.train()
            critic_opt.zero_grad()

            x, y_true = batch

            x = x.to(device)
            y_true = y_true.to(device)

            y_fake = model(x)

            # train critic phase
            batch_dim = x.shape[0]

            pred_true = critic(y_true)

            # forward pass on true
            loss_true = loss_discriminator(pred_true, torch.ones_like(pred_true))

            # then updates on fakes
            pred_fake = critic(y_fake.detach())
            loss_fake = loss_discriminator(pred_fake, torch.zeros_like(pred_fake))

            loss_discr = loss_true + loss_fake
            loss_discr *= 0.5

            loss_discr.backward()
            critic_opt.step()

            experiment.log_metric("d_loss_sum", loss_discr.item())
            experiment.log_metric("d_loss_real", 0.5 * loss_true.item())
            experiment.log_metric("d_loss_fake", 0.5 * loss_fake.item())
            experiment.log_metric("real_output", sigmoid(pred_true).mean().item())
            experiment.log_metric("fake_output", sigmoid(pred_fake).mean().item())

            ## train generator phase
            gan_opt.zero_grad()

            lpips_loss_ = lpips_loss(y_fake, y_true).mean()
            ssim_loss = 1.0 - ssim(y_fake, y_true)
            pred_fake = critic(y_fake)
            bce = loss_discriminator(pred_fake, torch.ones_like(pred_fake))
            loss_gen = w0 * lpips_loss_ + w1 * ssim_loss + l0 * bce

            loss_gen.backward()
            gan_opt.step()

            if step % 500 == 0:
                h, w = y_true.shape[2], y_true.shape[3]
                resized_x = F.interpolate(x.detach().clone(), size=(h,w))
                experiment.log_image(
                    make_grid(resized_x, y_true, y_fake), name="lq-hq-rec"
                )
                experiment.log_figure("g_grads", plot_grads(model))
                experiment.log_figure("d_grads", plot_grads(critic))


            experiment.log_metric("g_loss_sum", loss_gen.item())
            experiment.log_metric("g_loss_adv", l0 * bce.item())
            experiment.log_metric("g_loss_l1", w1 * ssim_loss.item())
            experiment.log_metric("g_loss_lpips", w0 * lpips_loss_.item())
            experiment.log_metric("fake_output'", sigmoid(pred_fake).mean().item())
            experiment.log_metric(
                "lpips gt-fake", lpips_alex(y_true, y_fake).mean().item()
            )

            content_loss = loss_gen.item() - l0 * bce.item()
            tqdm_.set_description(
                f"D loss: {loss_discr.item():.3}; Content loss: {content_loss:.3}; BCE/l0: {bce.item():.3}"
            )
            step += 1

        if (e + 1) % args.VALIDATION_FREQ == 0:
            print("Validation phase")

            ssim_validation = []
            lpips_validation = []

            tqdm_ = tqdm.tqdm(data_loader_eval)
            model.eval()
            for batch in tqdm_:
                x, y_true = batch
                with torch.no_grad():
                    x = x.to(device)
                    y_true = y_true.to(device)
                    y_fake = model(x)
                    ssim_val = ssim(y_fake, y_true).mean()
                    lpips_val = lpips_alex(y_fake, y_true).mean()
                    ssim_validation += [float(ssim_val)]
                    lpips_validation += [float(lpips_val)]

            ssim_mean = sum(ssim_validation) / len(ssim_validation)
            lpips_mean = sum(lpips_validation) / len(lpips_validation)

            print(f"Val SSIM: {ssim_mean}, Val LPIPS: {lpips_mean}")

            export_dir = str(args.EXPORT_DIR)
            if not os.path.exists(export_dir):
                os.mkdir(export_dir)
            torch.save(model.state_dict(),
                       export_dir + '/{0}_epoch{1}_ssim{2:.4f}_lpips{3:.4f}_crf{4}.pth'.format(arch_name, e, ssim_mean, lpips_mean,
                                                                                 args.CRF))

            # having critic's weights saved was not useful, better sparing storage!
            # torch.save(critic.state_dict(), 'critic_gan_{}.pth'.format(e + starting_epoch))
