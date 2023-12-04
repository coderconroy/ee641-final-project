from dataclasses import dataclass
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
from PIL import Image


def save_image_batch(image_batch, filename):
    """
    Save a batch of image tensors to disk as a single grid image

    Parameters:
    - image_batch: batch of unnormalized image tensors with shape (batch_size, channels, height, width)
    - filename: File path and name to save image without extension 
    """
    # Normalize images to [0, 1]
    image_batch = torch.clamp(image_batch * 0.5 + 0.5, 0, 1)

    # Create image grid and convert to PIL image
    image_grid = make_grid(image_batch)
    image_grid = np.transpose(image_grid.cpu().numpy(), (1, 2, 0))
    pil_image = Image.fromarray((image_grid * 255).astype(np.uint8))

    # Split the path into directory and filename
    path = f"{filename}.png"
    directory, filename = os.path.split(path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save image to disk
    pil_image.save(path)


def save_checkpoint(state, filename):
    # Split the path into directory and filename
    path = f"{filename}.pt"
    directory, filename = os.path.split(path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save checkpoint state
    torch.save(state, path)


def load_model_eval(checkpoint, model, diffusion):
    model.load_state_dict(checkpoint['state_dict'])
    diffusion.load_state_dict(checkpoint['diffusion'])
    return checkpoint['loss_history']


def get_dataloader(data_dir, batch_size, image_size):
    # Define image transformations
    # TODO: Review if this can be improved.
    transform = transforms.Compose([
        transforms.Resize(round(image_size * 5/4)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize from [0, 1] to [-1, 1]
    ])

    # Create an instance of the ImageFolder dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader to batch and shuffle the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader