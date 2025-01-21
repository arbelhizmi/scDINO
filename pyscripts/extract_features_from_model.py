import sys
import torch
import os
import argparse
from argparse import Namespace
import numpy as np
import torch.nn as nn
import tqdm
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms

from merfish_dataset import MerfishCellCenters
from vision_transformer import VisionTransformer
import vision_transformer as vits
import utils


def save_features_and_labels(
    features: torch.Tensor, labels: torch.Tensor, base_path_to_save_features_and_labels: Path,
) -> None:
    """
    Function to save features and labels (of cell types) extracted from the model.
    :param features: features as tensor
    :param labels: labels as tensor
    :param base_path_to_save_features_and_labels:
    """
    print(f'Shape of features: {features.shape}')
    print(f'Shape of labels: {labels.shape}')

    features_save_path = f'{base_path_to_save_features_and_labels}/features.pth'
    labels_save_path = f'{base_path_to_save_features_and_labels}/labels.pth'

    torch.save(features, features_save_path)
    torch.save(labels, labels_save_path)

    print(f'Features and labels were saved at: {base_path_to_save_features_and_labels}')


@torch.no_grad()
def extract_features(
    arguments: Namespace, model: VisionTransformer, data_loader: DataLoader, text_for_tqdm: str,
    gpu_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to get features (predictions) and labels from model.
    :param arguments: arguments
    :param model: model to get predictions from
    :param data_loader: data loader that contain input data (inputs and labels extracted from iterating the data loader)
    :param text_for_tqdm: text to print while running with tqdm progress bar
    :param gpu_device: torch device to have the model on
    :return: tuple of tensors represent the features and the labels
    """
    features_as_batches = []
    labels_as_batches = []

    transformations = transforms.Compose([])

    mean_for_selected_channel = arguments.norm_per_channel[0]
    std_for_selected_channel = arguments.norm_per_channel[1]

    # Defining scaling factor to resize the image to match the ratio of micron per pixel like scDINO dataset
    scale_factor = 1 / 0.6
    size_to_resize = arguments.crop_size * scale_factor
    assert int(size_to_resize) == size_to_resize
    transformations.transforms.append(transforms.Resize(int(size_to_resize)))

    # Resize to size of 224X224 like in scDINO paper
    transformations.transforms.append(transforms.Resize(arguments.resize_length))

    # Add channel-wise normalization using mean and std of each channel like scDINO paper
    transformations.transforms.append(
        transforms.Normalize(
            mean=mean_for_selected_channel, std=std_for_selected_channel
        ))

    for crops, cell_type, _ in tqdm.tqdm(data_loader, desc=f'{text_for_tqdm}', total=len(data_loader)):
        crops = crops.to(gpu_device, non_blocking=True)

        normalized_crop = transformations(crops)

        embeddings = model(normalized_crop)  # output dimension: (batch, embed_dim (e.g. 384))

        features_as_batches.append(embeddings.cpu())
        labels_as_batches.append(cell_type)

    print('After loop of features extraction')

    features_tensor = torch.cat(features_as_batches, dim=0)
    labels_tensor = torch.cat(labels_as_batches, dim=0)

    normalized_features = nn.functional.normalize(features_tensor, dim=1, p=2)

    return normalized_features, labels_tensor


def load_model(arguments: Namespace, gpu_device: torch.device) -> VisionTransformer:
    checkpoints = torch.load(arguments.model_path, map_location='cpu')

    sc_dino_args = checkpoints['args']

    # Remove the 'module' prefix from the keys
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoints[arguments.checkpoint_key].items()}

    # Remove the 'backbone', 'head' prefixes from the keys
    new_state_dict = {k.replace('backbone.', '').replace('head.', ''): v for k, v in new_state_dict.items()}
    # new_state_dict = {k.replace('backbone.', ''): v for k, v in new_state_dict.items()}

    vit_small_params = {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
    }

    # model = VisionTransformer(
    #     patch_size=sc_dino_args.patch_size,
    #     drop_path_rate=sc_dino_args.drop_path_rate,
    #     in_chans=len(arguments.selected_channels),
    #     num_classes=0,
    #     img_size=[arguments.resize_length, ],
    #     **dict(vit_small_params),
    # )
    # Create the model instance
    model = vits.__dict__[sc_dino_args.arch](
        patch_size=arguments.patch_size,
        in_chans=len(arguments.selected_channels),
        num_classes=0,
        img_size=[arguments.resize_length, ],
    )

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(gpu_device)

    return model


def extract_features_pipeline(arguments: Namespace) -> None:
    local_rank = torch.cuda.current_device()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(arguments)).items())))

    device = torch.device(f'cuda:{local_rank}')

    # Load model
    model = load_model(arguments, device)

    # Load Merfish dataset
    merfish_cell_centers_dataset = MerfishCellCenters(
        arguments.liver_slices_directory_path, arguments.liver_slice_names, arguments.transformed_coordinates_csv_path,
        arguments.crop_size, arguments.local_crops_number, transforms=None,
    )

    data_loader = torch.utils.data.DataLoader(
        merfish_cell_centers_dataset,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    text_for_tqdm = f'Extracting features'

    # Define save path
    if arguments.unified:
        features_directory_name = f'features_and_labels/unified/features_and_labels_{arguments.checkpoint_key}'
    else:
        features_directory_name = f'features_and_labels/not_unified/features_and_labels_{arguments.checkpoint_key}'

    save_base_path = Path(
        f'{arguments.transformed_coordinates_csv_path}/{arguments.liver_slice_names[0]}/{features_directory_name}'
    )
    Path(save_base_path).mkdir(parents=True, exist_ok=True)

    # Extract and save features
    features, labels = extract_features(arguments, model, data_loader, text_for_tqdm, device)

    save_features_and_labels(features, labels, save_base_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Feature extraction')
    # computation settings
    parser.add_argument(
        '--model_path', type=str,
        default='/home/labs/nyosef/Collaboration/Arbel/vizgen_liver/scDINO_checkpoints/sc-ViT_checkpoint0100_vitsmall16.pth',
    )
    parser.add_argument(
        "--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")'
    )
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size', default=1024, type=int, help='Per-GPU batch-size')

    # Dataset arguments
    parser.add_argument(
        '--liver_slices_directory_path', type=str, default='/home/labs/nyosef/jonasm/vizgen',
        help='path to the liver slices directory (e.g. like ../jonasm/vizgen directory)',
    )
    parser.add_argument(
        '--liver_slice_names', default=['Liver1Slice1', ], type=list, nargs='+',
        help='list of liver slice names as list of strs (like Liver1Slice1 etc.)',
    )
    parser.add_argument(
        '--transformed_coordinates_csv_path', type=str, default='/home/labs/nyosef/Collaboration/Arbel/vizgen_liver',
        help='path to the local liver slices directory (e.g. like ../Collaboration/Arbel/vizgen_liver directory)',
    )
    parser.add_argument(
        '--unified', type=utils.bool_flag, default=True,
        help="flag to indicate whether to use 'Hepatocyte' as a single cell type (all cell types which are versions of "
             "'Hepatocyte' are unified to a single cell type (for example, 'Hepatocyte, peri-portal' would just"
             "be 'Hepatocyte'"
    )

    parser.add_argument(
        '--crop_size', type=int, default=36, help='crop size around cell centers in pixels (should be even number)',
        # '--crop_size', type=int, default=32, help='crop size around cell centers in pixels (should be even number)',
    )
    parser.add_argument('--resize_length', default=224, type=int, help="""quadratic resize length to resize images""")
    parser.add_argument(
        '--local_crops_number', type=int, default=8, help="""Number of small local views to generate. Set this parameter
         to 0 to disable multi-crop training. When disabling multi-crop we recommend to use
         "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument(
        '--selected_channels', default=[0, 1, 2, 3, 4], type=list,
        help="""list of channel indexes of the .tiff images which should be used to create the tensors."""
    )
    parser.add_argument(
        '--norm_per_channel', help="""2x tuple of mean and std per channel typically values between 0 and 1""",
        default=[(0.2981, 0.5047, 0.1918, 0.2809, 0.3800), (0.2232, 0.1659, 0.2403, 0.1938, 0.1956)], type=list,
    )

    args = parser.parse_args()

    extract_features_pipeline(args)
