from pathlib import Path
from typing import List, Dict
import numpy as np
import zarr


# FILTERED_COORDINATES_CSV_NAME = 'filtered_transformed_cell_metadata.csv'
FILTERED_COORDINATES_CSV_NAME = 'cleaned_unified_annotated_cell_metadata.csv'


def check_if_zarr_image_exists_and_not_empty(zarr_image_liver_directory: Path) -> None:
    """
    Function to check if path to images.zarr is exists and not empty
    :param zarr_image_liver_directory: path to where liver-slice (in Jonas's directory)
    """
    path_of_zarr_images_of_slice = zarr_image_liver_directory / f'images/rescaled0.5um/images.zarr/'

    # Check if the path to the images.zarr directory exists. If not, raise error
    if not path_of_zarr_images_of_slice.is_dir():
        raise NotADirectoryError(f"Path to images.zarr wasn't found: {str(path_of_zarr_images_of_slice)}")

    # If path to zarr images is an empty directory
    if not any(path_of_zarr_images_of_slice.iterdir()):
        raise FileNotFoundError(f'images.zarr directory: {path_of_zarr_images_of_slice} is empty')

    return


def check_if_transformed_coordinates_csv_exists(transformed_coordinates_liver_directory: Path) -> None:
    """
    Function to check if filtered_transformed_cell_metadata.csv exists in the current liver-slice directory
    :param transformed_coordinates_liver_directory: path to liver directory (in Collaboration)
    """
    path_of_transformed_coordinates_of_slice = transformed_coordinates_liver_directory / FILTERED_COORDINATES_CSV_NAME

    # Check if the file ../<liver-slice-name>/filtered_transformed_cell_metadata.csv exists. If not, raise error
    if not path_of_transformed_coordinates_of_slice.exists():
        raise FileNotFoundError(
            f"transformed_cell_metadata.csv {path_of_transformed_coordinates_of_slice} doesn't exists"
        )

    return


def align_paths_with_slice_names(
    liver_slice_names: List[str], dictionary_of_paths: [str, List[str]],
) -> Dict[str, List[str]]:
    """
    Function to align and sort paths according to the order of live slice names list. Given a list of liver slice names
    and a dictionary containing lists of paths, reorder each list of paths, so they align with the order
    of liver_slice_names
    :param liver_slice_names: list of liver slice names
    :param dictionary_of_paths: paths to align and sort
    :return: sorted dictionary_of_paths dictionary
    """
    # Create a mapping from slice name to index
    slice_to_index = {name: idx for idx, name in enumerate(liver_slice_names)}

    def get_slice_order_from_path(path: str) -> int:
        for name in liver_slice_names:
            if name in path:
                return slice_to_index[name]
        raise ValueError(f"No known liver slice name found in path: {path}")

    # Create a copy of dictionary_of_paths to avoid mutating the original
    sorted_dictionary_of_paths = {}
    for key, paths_list in dictionary_of_paths.items():
        # Sort the paths list in-place according to slice order
        sorted_list = sorted(paths_list, key=get_slice_order_from_path)
        sorted_dictionary_of_paths[key] = sorted_list

    return sorted_dictionary_of_paths


def find_zarr_images_and_transformed_coordinates_paths(
    liver_slices_directory_path: Path, liver_slice_names: List[str], slices_transformed_coordinates_base_path: Path,
) -> Dict[str, List[str]]:
    """
    Function to find the paths to the images.zarr folders and to the filtered_transformed_cell_metadata.csv
    of each liver-slice.
    The function also get the paths to the file contain the mean and std of *each channel* in slice
    :param liver_slices_directory_path: path to liver directories which contain the images.zarr files
    :param liver_slice_names: list of names of slices (e.g. Liver1Slice1)
    :param slices_transformed_coordinates_base_path: path to liver folders which contain transformed coordinates files
    :return: dictionary of lists with the paths (paths already aligned and lists should be in the same length)
    """
    # Liver slices directory path is
    if not liver_slices_directory_path.is_dir():
        raise NotADirectoryError(f"Liver slices path is not a directory: {str(liver_slices_directory_path)}")

    zarr_images_of_slices_paths = []
    transformed_coordinates_of_slices_paths = []
    mean_and_std_per_channel_of_slice_paths = []

    # Iterating the liver-slices names (liver slice name could be 'Liver1SLice1', for example)
    for liver_slice_name in liver_slice_names:

        # Path to liver-slice (like ../jonasm/vizgen/Liver1Slice1)
        liver_directory_of_zarr_image = liver_slices_directory_path / liver_slice_name
        # path_of_zarr_images_of_slice = liver_directory_of_zarr_image / f'images/rescaled0.5um/images.zarr/'

        # Check if images.zarr directory exists and not empty
        check_if_zarr_image_exists_and_not_empty(liver_directory_of_zarr_image)

        # Path to liver-slice of transformed coordinates (like in Collaboration/Arbel/vizgen_liver folder)
        liver_directory_of_transformed_coordinates = slices_transformed_coordinates_base_path / liver_slice_name
        # path_of_transformed_coordinates_of_slice = liver_directory_of_transformed_coordinates / 'FILTERED_COORDINATES_CSV_NAME'
        check_if_transformed_coordinates_csv_exists(liver_directory_of_transformed_coordinates)

        zarr_images_of_slices_paths.append(str(liver_directory_of_zarr_image / f'images/rescaled0.5um/images.zarr/'))
        transformed_coordinates_of_slices_paths.append(
            # str(liver_directory_of_transformed_coordinates / f'FILTERED_COORDINATES_CSV_NAME')
            str(liver_directory_of_transformed_coordinates / FILTERED_COORDINATES_CSV_NAME)
        )
        mean_and_std_per_channel_of_slice_paths.append(
            str(liver_directory_of_transformed_coordinates / 'slice_mean_and_std.json')
        )

    number_of_zarrs = len(zarr_images_of_slices_paths)
    number_of_transformed = len(transformed_coordinates_of_slices_paths)
    if number_of_transformed != number_of_zarrs:
        raise ValueError(
            f'Number of zarr images paths != number of transformed_cell_metadata csv files:'
            f'number of zarr paths: {number_of_zarrs}. Number of transformed paths: {number_of_transformed}'
        )

    dictionary_of_paths = {
        'zarr_images_paths': zarr_images_of_slices_paths,
        'transformed_coordinates_paths': transformed_coordinates_of_slices_paths,
        'mean_and_std_paths': mean_and_std_per_channel_of_slice_paths,
    }

    sorted_dictionary_of_paths = align_paths_with_slice_names(liver_slice_names, dictionary_of_paths)

    return sorted_dictionary_of_paths


def extract_patch(zarr_image: zarr.array, x_center: float, y_center: float, patch_size: int) -> zarr.array:
    x_center = int(round(x_center))
    y_center = int(round(y_center))

    half_patch = patch_size // 2

    y_start = y_center - half_patch
    y_end = y_center + half_patch
    x_start = x_center - half_patch
    x_end = x_center + half_patch

    height, width = zarr_image.shape[-2], zarr_image.shape[-1]
    z_mid = zarr_image.shape[1] // 2

    # Check if patch is within image boundaries
    if y_start < 0 or x_start < 0 or y_end > height or x_end > width:
        # Patch goes out of the image boundary
        return None

    # Extract the patch
    patch = zarr_image[:, z_mid, y_start:y_end, x_start:x_end].astype(np.float32)

    return patch


class PatchExceedImageBoundariesException(Exception):
    pass
