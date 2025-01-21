from torch.utils.data import Dataset
from typing import Dict, List
import pandas as pd
import zarr
import json
from sklearn.preprocessing import LabelEncoder

from merfish_dataset_utils import extract_patch, PatchExceedImageBoundariesException


class MerfishSingleLiverSlice(Dataset):
    """
    A dataset for a single FOV
    """
    def __init__(
        self, liver_slice_name: str, zarr_images_directory: str, transformed_coordinates_file_path: str,
        mean_and_std_per_channel_file_path: str, patch_size: int,
    ) -> None:
        super(MerfishSingleLiverSlice, self).__init__()
        self.liver_slice_name = liver_slice_name
        self.zarr_image = self.read_zarr_image(zarr_images_directory)
        self.transformed_cell_metadata = self.read_transformed_cell_metadata(transformed_coordinates_file_path)
        self.mean_and_std_of_channels = self.read_mean_and_std_per_channel(mean_and_std_per_channel_file_path)
        self.patch_size = patch_size
        self.labels = self._set_labels_for_dataset()

    def __getitem__(self, idx: int):
        # Retrieve the cell coordinates from the DataFrame
        cell = self.transformed_cell_metadata.iloc[idx]
        x_center = cell['pixel_x']
        y_center = cell['pixel_y']

        # Extract patch from the zarr image
        patch = extract_patch(self.zarr_image, x_center, y_center, self.patch_size)

        cell_type = cell['annotation']

        if patch is None:
            raise PatchExceedImageBoundariesException(
                f'Patch at index: {idx} with cell center (x, y): {(x_center, y_center)} exceeding image boundaries'
            )

        return patch, cell_type, self.liver_slice_name

    def __len__(self):
        return self.transformed_cell_metadata.shape[0]

    @staticmethod
    def read_transformed_cell_metadata(transformed_coordinates_file_path: str) -> pd.DataFrame:
        cell_metadata = pd.read_csv(transformed_coordinates_file_path, index_col=0, header=0, nrows=None)
        return cell_metadata

    @staticmethod
    def read_zarr_image(zarr_images_directory: str) -> zarr.array:
        zarr_image = zarr.open(zarr_images_directory, mode='r')
        return zarr_image

    @staticmethod
    def read_mean_and_std_per_channel(mean_and_std_per_channel_file_path: str) -> Dict[str, List[float]]:
        # Load the JSON file
        with open(mean_and_std_per_channel_file_path, 'r') as file:
            mean_and_std_of_channels = json.load(file)

        channel_stats = [[0.2981, 0.5047, 0.1918, 0.2809, 0.3800], [0.2232, 0.1659, 0.2403, 0.1938, 0.1956]]

        means_and_stds = {
            "mean": channel_stats[0],
            "std": channel_stats[1],
        }

        # return mean_and_std_of_channels
        return means_and_stds

    def _set_labels_for_dataset(self):
        labels = self.transformed_cell_metadata['annotation'].unique()

        # Create the encoder
        label_encoder = LabelEncoder()

        # Fit and transform the labels
        numeric_labels = label_encoder.fit_transform(labels)

        # Mapping of string to numeric labels
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

        # return numeric_labels
        return label_mapping
