from typing import List, Dict
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path

from dino_utils import DataAugmentationDINO
from merfish_dataset_utils import find_zarr_images_and_transformed_coordinates_paths
from single_liver_slice_dataset import MerfishSingleLiverSlice


class MerfishCellCenters(Dataset):
    def __init__(
        self, liver_slices_directory_path: str, liver_slice_names: List[str], transformed_coordinates_csv_path: str,
        crop_size: int, number_of_crops: int, transforms: DataAugmentationDINO = None,
    ) -> None:
        # Paths in dictionary are ordered and aligned to the same order in liver_slice_names
        zarr_images_and_transformed_coordinates_and_mean_std_paths = find_zarr_images_and_transformed_coordinates_paths(
            Path(liver_slices_directory_path), liver_slice_names, Path(transformed_coordinates_csv_path)
        )
        self.liver_slice_names = liver_slice_names
        self.crop_size = crop_size
        self.liver_slices = ConcatDataset(
            self._create_a_single_live_slice_object(zarr_images_and_transformed_coordinates_and_mean_std_paths)
        )
        self.cumulative_lengths_of_liver_datasets = self._get_cumulative_lengths()
        self.transforms = transforms

        # self.in_channels = len(channel_groups[0])  # number of input channels for model

    def __len__(self) -> int:
        return self.cumulative_lengths_of_liver_datasets[-1]  # Total number of samples

    def __getitem__(self, idx: int):
        # Determine which sub-dataset the index belongs to
        for i, cumulative_length in enumerate(self.cumulative_lengths_of_liver_datasets):
            if idx < cumulative_length:
                # Found the sub-dataset
                liver_slice = self.liver_slices.datasets[i]

                # Calculate the local index
                local_idx = idx if i == 0 else idx - self.cumulative_lengths_of_liver_datasets[i - 1]

                # Fetch the item from the sub-dataset
                crop, cell_type, liver_slice_name = liver_slice[local_idx]

                # Convert cell type to numerical label
                cell_type_numerical_label = liver_slice.labels[cell_type]

                if self.transforms is not None:
                    crop = self.transforms(crop, liver_slice.mean_and_std_of_channels)

                return crop, cell_type_numerical_label, liver_slice_name

        raise IndexError("Index out of range in __getitem__ function of MerfishCellCenters dataset")

    def _create_a_single_live_slice_object(
        self, paths_of_liver_slices: Dict[str, List[str]],
    ) -> List[MerfishSingleLiverSlice]:
        """
        Function to create list of live-slices objects to be used as the dataset
        :param paths_of_liver_slices: dictionary of paths to liver slices and their metadata
        :return: list of single liver slice objects
        """
        liver_slices_dataset = []

        # Now we can iterate easily:
        for index, liver_slice_name in enumerate(self.liver_slice_names):
            zarr_image_path = paths_of_liver_slices['zarr_images_paths'][index]
            transformed_coordinates_path = paths_of_liver_slices['transformed_coordinates_paths'][index]
            mean_std_path = paths_of_liver_slices['mean_and_std_paths'][index]
            merfish_liver_slice = MerfishSingleLiverSlice(
                liver_slice_name=liver_slice_name,
                zarr_images_directory=zarr_image_path,
                transformed_coordinates_file_path=transformed_coordinates_path,
                mean_and_std_per_channel_file_path=mean_std_path,
                patch_size=self.crop_size,
            )
            liver_slices_dataset.append(merfish_liver_slice)

        return liver_slices_dataset

    def _get_cumulative_lengths(self) -> List[int]:
        cumulative_lengths = []
        total = 0
        for liver_slice in self.liver_slices.datasets:
            total += len(liver_slice)
            cumulative_lengths.append(total)

        return cumulative_lengths
