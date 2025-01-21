from torchvision import transforms
import torch
# from pyscripts import utils
import utils


class DataAugmentationDINO(object):
    def __init__(
        self, images_are_RGB, global_crops_scale, local_crops_scale, local_crops_number,
    ):

        if not images_are_RGB:
            flip_gamma_brightness = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # utils.AdjustGamma_custom(0.8),
                utils.AdjustBrightness(0.8),
            ])
            # normalize = transforms.Compose([
            #     # utils.normalize_0_to_1(),
            #     transforms.Normalize(mean_for_selected_channel, std_for_selected_channel),
            # ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(1.0),
                # normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(0.1),
                utils.Solarization_forGreyscaleMultiChan(0.2),
                # normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(0.5),
                # normalize,
            ])

        # images are RGB
        else:
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
            ])
            # normalize = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean_for_selected_channel, std_for_selected_channel),
            # ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                # normalize,
                to_tensor,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                # normalize,
                to_tensor,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale,
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                # normalize,
                to_tensor,
            ])

    def __call__(self, image, liver_slice_mean_and_std):
        # Extract the mean and std of channels each liver slice
        mean_of_slice_channels = liver_slice_mean_and_std['mean']
        std_of_slice_channels = liver_slice_mean_and_std['std']

        # Create the normalization transformation
        normalization_transformation = transforms.Normalize(mean=mean_of_slice_channels, std=std_of_slice_channels)

        crops = []
        # crops.append(self.global_transfo1(image))
        # crops.append(self.global_transfo2(image))

        # Global crops
        global_crop_1 = self.global_transfo1(image)
        global_crop_2 = self.global_transfo2(image)

        # Apply normalization to global crops
        normalized_global_crop_1 = normalization_transformation(global_crop_1)
        normalized_global_crop_2 = normalization_transformation(global_crop_2)

        crops.append(normalized_global_crop_1)
        crops.append(normalized_global_crop_2)

        for _ in range(self.local_crops_number):
            local_crop = self.local_transfo(image)

            normalized_local_crop = normalization_transformation(local_crop)

            crops.append(normalized_local_crop)
            # crops.append(self.local_transfo(image))
        for crop in crops:
            if torch.isnan(crop).any():
                print("Nan found in the crop")
        return crops


