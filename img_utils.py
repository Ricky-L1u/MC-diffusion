import torch
import torch.nn.functional as F
from torch.nn.functional import fold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import imageio
import math
from skin_render import render_3d_skin
import io

_h, _w = 64, 64
topil = transforms.ToPILImage()
totensor = transforms.ToTensor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mask(ref_img, mask_img):
    return (ref_img != mask_img).any(dim=0, keepdims=True).repeat(4, 1, 1).float()  # Any better way?


def get_masked_img(mask, ref_img, mask_img):
    return (ref_img * (1 - mask)) + (mask_img * mask)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img


def tensor_to_img(tensor, binary_alpha=True):
    out = unflatten(tensor)
    out = out.clip(-1, 1)
    if binary_alpha:
        out[:, -1, :, :][out[:, -1, :, :] > 0] = 1
        out[:, -1, :, :][out[:, -1, :, :] <= 0] = -1
    return out.squeeze(1) / 2 + 0.5


def save_img_to(path, tensor,binary_alpha = True):
    imgs = tensor_to_img(tensor,binary_alpha = binary_alpha)
    for index, i in enumerate(imgs):
        img = topil(i)
        img.save(f'{path}/{index}.png')


def display_img_from_tensor(tensor, binary_alpha=True, render=True, display=True, gif=False):
    return display_image(tensor_to_img(tensor, binary_alpha=binary_alpha), render=render, display=display, gif=gif)


def img_from_path(path, device):
    return (totensor(Image.open(path).convert('RGBA')).to(device) - 0.5) * 2


def tensor_from_path(path, device, n_samples=1):
    return (flatten(totensor(Image.open(path).convert('RGBA'))).to(device).unsqueeze(
        0).repeat(n_samples, 1, 1) - 0.5) * 2


def filter_img(x):
    return flatten_batch(low_pass_filter(unflatten(x)))  # Pretty inefficient but it works


def render_skin(img, rotation=35):
    # async def render_skin():
    #     # Render a full body skin
    return render_3d_skin(skin_image=img, hr=rotation, aa=False, vr=-15)

    # return asyncio.run(render_skin())


def flatten(x, to_flatten=True, patch_size=4):
    if to_flatten:
        v = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        x = v.permute(1, 2, 0, 3, 4)
        return x.flatten(2, -1).flatten(0, 1)
    return x


def flatten_batch(x, to_flatten=True, patch_size=4):
    if to_flatten:
        v = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = v.permute(0, 2, 3, 1, 4, 5)
        return x.flatten(3, -1).flatten(1, 2)
    return x


def unflatten(x):
    return fold(x.permute(0, 2, 1), (64, 64), (4, 4), stride=4)


def low_pass_filter(x, n=4):
    return F.interpolate(F.avg_pool2d(x, n), scale_factor=n)


def filter_img(x):
    return flatten_batch(low_pass_filter(unflatten(x)))  # Pretty inefficient but it works


class SkinLoader(Dataset):
    def __init__(self, root_dir, patch_size=4, to_flatten=True):
        self.root_dir = root_dir
        self.totensor = transforms.ToTensor()
        self.patch_size = patch_size
        self.to_flatten = to_flatten

    def __getitem__(self, idx):
        return (flatten(self.totensor(Image.open(f"{self.root_dir}\\{idx}.png").convert('RGBA')), self.to_flatten,
                        self.patch_size) - 0.5) * 2

    def __len__(self):
        return len(os.listdir(self.root_dir)) - 1


def display_image(images, render=True, display=True, gif=False):
    if not (gif or display):
        return
    topil = transforms.ToPILImage()
    images = [topil(i) for i in images]
    fig = plt.figure(figsize=(8, 8))
    img = None
    for index, image in enumerate(images):
        fig.add_subplot(math.ceil(len(images) ** 0.5), math.ceil(len(images) ** 0.5), index + 1)
        if render:
            plt.imshow(render_skin(image))
        else:
            plt.imshow(image)
    for ax in fig.axes:
        ax.axis('off')
        ax.patch.set_facecolor('none')

    if display:
        plt.show()
    if gif:
        img = fig2img(fig)
    plt.close(fig)
    return img


def img_to_gif(images, path):
    imageio.mimsave(path, images, fps=30)
