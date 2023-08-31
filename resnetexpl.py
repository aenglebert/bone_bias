import argparse
import os
import time
import sys

import numpy as np
import torch

import pytorch_lightning as pl

from torchvision import transforms

from models.xraymodels import Resnet50
from models.utils import UnList

from tqdm import tqdm

from datasets.mura import MuraDataModule, MURADataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import captum
from captum.attr import visualization as viz
from captum.metrics import infidelity, sensitivity_max

from PolyCAM.polycam.polycam import PCAMp

parser = argparse.ArgumentParser(description="XAI on x-rays")

#########################
#### data parameters ####
#########################
parser.add_argument("--mura_data_dir", type=str, default='./input/mura-v11',
                    help="path to MURA repository")

#########################
##### model params ######
#########################
parser.add_argument("--checkpoint", default=None, type=str,
                    help="checkpoint file")


#########################
#### optim parameters ###
#########################
parser.add_argument("--batch_size", default=16, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")


#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")
parser.add_argument("--seed", type=int, default=31, help="seed. If value set 0 = random seed")


custom_red_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#000000'),
                                                          (0.2, '#880E4F'),
                                                          (1, '#E91E63')], N=256)
custom_blue_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#000000'),
                                                          (0.25, '#000000'),
                                                          (1, '#80DEEA')], N=256)


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.03, inputs.shape)).float().cuda()
    return noise, inputs - noise


class AttributionsBase():
    def __init__(self, attribute_fn, show, sign, alpha=0.2, cmap=None):
        self.attribute_fn = attribute_fn
        self.sign = sign
        self.show = show
        self.cmap = cmap
        self.alpha = alpha

    def get_visualisation(self, fig=None, ax=None):

        if fig is None or ax is None:
            plt_fig_axis = None
        else:
            plt_fig_axis = [fig, ax]
        _ = viz.visualize_image_attr(np.transpose(self.attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     np.transpose(self.x.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     cmap=self.cmap,
                                     method="blended_heat_map",
                                     alpha_overlay=self.alpha,
                                     show_colorbar=False,
                                     plt_fig_axis=plt_fig_axis,
                                     sign=self.sign,
                                     outlier_perc=2,
                                     use_pyplot=self.show)


class SaliencyAttributions(AttributionsBase):
    def __init__(self, attribute_fn, baseline=None, show=False, sign="all", nt_samples_batch_size=None, cam=False, cmap=None):
        super().__init__(attribute_fn, show, sign, cmap=cmap)
        if cmap is None:
            self.cmap = custom_blue_cmap
            self.cmap = custom_red_cmap
        self.alpha = 0.1
        self.nt_samples_batch_size = nt_samples_batch_size
        self.baseline = baseline
        self.cam = cam

    def __call__(self, x, *args, **kwargs):
        self.x = x
        if self.baseline == "gaussian_blur":
            x_blur = self.gaussian_blur()
            if self.nt_samples_batch_size:
                self.attribution = self.attribute_fn(x, baselines=x_blur, nt_samples_batch_size=self.nt_samples_batch_size)
            else:
                self.attribution = self.attribute_fn(x, baselines=x_blur)

        else:
            if self.nt_samples_batch_size:
                self.attribution = self.attribute_fn(x, nt_samples_batch_size=self.nt_samples_batch_size)
            else:
                self.attribution = self.attribute_fn(x)
        if self.cam:
            self.attribution = \
                torch.ones(self.x.shape).cuda() * captum.attr.LayerAttribution.interpolate(self.attribution, self.x.shape[-2:], interpolate_mode='bicubic')

        return self.attribution


    def gaussian_blur(self):
        s = 99
        return torchvision.transforms.functional.gaussian_blur(self.x, kernel_size=[s, s], sigma=20 )


class OcclusionAttributions(AttributionsBase):
    def __init__(self,
                 attribute_fn,
                 show=False,
                 sign="all",
                 perturbations_per_eval=16,
                 strides=(3, 16, 16),
                 sliding_window_shapes=(3, 32, 32)):
        super().__init__(attribute_fn, show, sign)
        self.perturbations_per_eval = perturbations_per_eval
        self.strides = strides
        self.sliding_window_shapes = sliding_window_shapes
        self.cmap = cm.get_cmap("jet")

    def __call__(self, x, *args, **kwargs):
        self.x = x
        self.attribution = self.attribute_fn(x,
                                           perturbations_per_eval=16,
                                           strides=(3, 16, 16),
                                           sliding_window_shapes=(3, 32, 32))
        return self.attribution


def main():
    global args
    args = parser.parse_args()

    assert args.checkpoint is not None, "Please specify a checkpoint to load with option --checkpoint"

    #Logger
    #writer = SummaryWriter(log_dir='./visualisation/' + args.checkpoint)

    # set seeds
    seed = args.seed
    # if args.seed is zero, select a random seed
    if args.seed == 0:
        seed = np.random.randint(4294967295)
    pl.seed_everything(seed)

    torch.multiprocessing.set_sharing_strategy('file_system')

    input_size = 320

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    ])

    # load checkpoint and get encoder
    model = Resnet50.load_from_checkpoint(args.checkpoint)
    encoder = model.encoder

    # output directory
    output_dir = os.path.join("visualisations", args.checkpoint.split("/")[-1].split(".")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imax = 200

    model.set_hooks()

    test_csv = os.path.join(args.mura_data_dir, "MURA-v1.1", "valid_labeled_studies.csv")
    mura_test = MURADataset(args.mura_data_dir, test_csv, transform=transform, study_level=True)

    test_dl = DataLoader(mura_test, batch_size=1, num_workers=args.workers)

    attributions_methods = {}
    visualisations_methods = {}

    #ixg = captum.attr.InputXGradient(model)
    #attributions_methods["Input x Grad"] = SaliencyAttributions(ixg.attribute, sign="positive")

    jet_cmap = cm.get_cmap("jet")

    #nt_ixg = captum.attr.NoiseTunnel(ixg)
    #attributions_methods["Input x Grad_SG"] = SaliencyAttributions(nt_ixg.attribute, sign="positive")

    #gbp = captum.attr.GuidedBackprop(model)
    #attributions_methods.append(gbp)
    #attributions_methods.append(captum.attr.NoiseTunnel(gbp))

    #ig = captum.attr.IntegratedGradients(model)
    #attributions_methods["Integrated Gradients"] = SaliencyAttributions(ig.attribute,
    #                                                                    sign="positive")

    #ig_blur = captum.attr.IntegratedGradients(model)
    #attributions_methods["Integrated Grad baseline blur"] = SaliencyAttributions(ig_blur.attribute,
    #                                                                    sign="positive",
    #                                                                    baseline="gaussian_blur")

    #nt_ig = captum.attr.NoiseTunnel(ig)
    #attributions_methods["Integrated Gradients_SG"] = SaliencyAttributions(nt_ig.attribute,
    #                                                                       sign="positive",
    #                                                                       baseline="gaussian_blur",
    #                                                                       nt_samples_batch_size=1)

    occ = captum.attr.Occlusion(model)
    attributions_methods["Occlusion"] = OcclusionAttributions(occ.attribute, sign="positive")

    pcam_fn = PCAMp(model.encoder, batch_size=args.batch_size, target_layer_list=["relu", "layer1", "layer2", "layer3", "layer4"])

    custom_blue_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#84ffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

    with torch.no_grad():
        model = model.cuda()
        for i, batch in enumerate(tqdm(test_dl)):
            if i >= imax:
                return
            x_list, y = batch
            images_list = []
            fig, ax = plt.subplots(len(x_list), len(attributions_methods)+3,
                                   figsize=((len(attributions_methods)+3)*4, len(x_list)*4))
            # Check if 2d and a dim if needed
            if type(ax[0]) != np.ndarray:
                ax = np.expand_dims(ax, axis=0)
            for j, x in enumerate(x_list):
                x = x.cuda()
                result = model(x).detach().cpu()
                pil_image = transforms.functional.to_pil_image(x[0].cpu())
                cam = model.cam()
                cam_image = transforms.ToTensor()(overlay(x[0], cam[0]))
                ax[j][0].imshow(pil_image)
                ax[j][0].get_xaxis().set_visible(False)
                ax[j][0].yaxis.set_major_locator(plt.NullLocator())
                ax[j][0].xaxis.set_major_formatter(plt.NullFormatter())
                ax[0][0].set_title("X-Ray")
                ax[j][0].set_ylabel("{:.2f}".format(result.item()), size="large")
                ax[j][1].imshow(overlay(x[0], cam[0]))
                ax[j][1].get_xaxis().set_visible(False)
                ax[j][1].get_yaxis().set_visible(False)
                ax[0][1].set_title("CAM")

                pcam = pcam_fn(x)[-1]
                ax[j][2].imshow(overlay(x[0], pcam[0]))
                ax[j][2].get_xaxis().set_visible(False)
                ax[j][2].get_yaxis().set_visible(False)
                ax[0][2].set_title("PCAM")


                #dscam = DeepScoreCAM(model, batch_size=4)
                #deepcam = dscam(x)
                #ax[j][2].imshow(overlay(x[0], deepcam[0]))
                #ax[j][2].get_xaxis().set_visible(False)
                #ax[j][2].get_yaxis().set_visible(False)
                #ax[0][2].set_title("DeepCAM")

                for count, method in enumerate(attributions_methods.__iter__()):
                    # attributions
                    attr = attributions_methods[method](x)
                    # get visualisation inside the plot
                    #print(method)
                    attributions_methods[method].get_visualisation(fig, ax[j][count+3])
                    ax[0][count+3].set_title(method)
                    #compute metrics
                    #infid = infidelity(model, perturb_fn, x, attr)
                    #sens = sensitivity_max(attributions_methods[method], x, max_examples_per_batch=1)
                    #sens = torch.tensor(0)
                    # show metrics
                    #ax[j][count + 2].yaxis.set_major_locator(plt.NullLocator())
                    #ax[j][count + 2].xaxis.set_major_formatter(plt.NullFormatter())
                    #ax[j][count + 2].set_xlabel("inf {:.5f}, se {:.5f}".format(infid.item(), sens.item()), size="x-small")
                    #ax[j][count + 2].set_xlabel("infidelity {:.5f}".format(infid.item()), size="x-small")


            fig.tight_layout(pad=1, w_pad=0.5, h_pad=1)
            #plt.show()
            plt.savefig(output_dir + "/" + str(i) + "_label" + str(y.item()) + ".png")

            #writer.add_figure(str(i) + "_label" + str(y.item()), fig)


def overlay(input, cam, alpha=0.7, colormap="jet"):
    # inspired by https://github.com/frgfm/torch-cam/blob/master/torchcam/utils.py

    if input.dim() != 3:
        raise AssertionError("input dimension should be 3, received " + input.dims())

    img = transforms.ToPILImage()(input)
    # normalize to 0,1
    cam -= torch.min(cam)
    cam /= torch.max(cam)
    cam_img = transforms.ToPILImage(mode='F')(cam)

    if type(colormap) is str:
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Resize mask and apply colormap
    overlay_raw = cam_img.resize(img.size, resample=Image.BICUBIC)
    overlay = overlay_raw
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
    return overlayed_img


def outlier_norm(intensor, percent=2, sign="all"):
    oned = intensor.reshape(-1)
    size = oned.shape[0]
    sort_tensor, _ = torch.sort(oned)
    min_clip_value = sort_tensor[int(size*percent/200)].item()
    max_clip_value = sort_tensor[int(size-size*percent/200)].item()
    if sign == "positive":
        min_clip_value = 0
    elif sign == "negative":
        max_clip_value = 0
    clamped = torch.clamp(intensor, min=min_clip_value, max=max_clip_value)
    if sign == "absolute":
        return torch.abs(clamped)
    else:
        return clamped


def grouped_overlay(batch, batch_cam, seq_sizes=None):
    if seq_sizes is None:
        seq_sizes = torch.ones(batch.shape[0])

    n_studies = len(seq_sizes)

    n_features_per_image = features.shape[-1] * features.shape[-2]
    features_size = features.shape[-3]

    grouped_features = torch.zeros((n_studies, features_size, n_features_per_image * max_per_study)).cuda()

    cumulative = 0
    for i in range(len(seq_sizes)):
        cur_flat_size = n_features_per_image * seq_sizes[i]
        grouped_features[i, :, 0:cur_flat_size] \
            = features[cumulative:cumulative + seq_sizes[i]].swapaxes(0, 1).flatten(start_dim=1)
        cumulative += seq_sizes[i]

    return grouped_features


if __name__ == "__main__":
    main()
