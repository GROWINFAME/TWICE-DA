import albumentations as A
import numpy as np
import torch


class AlbumentationsRandAugment(A.DualTransform):
    def __init__(self,
                 N_TFMS: int = 2,  # Number of transformation in each composition.
                 p=1.0):
        super(AlbumentationsRandAugment, self).__init__(p)
        self.transform_list = self.albumentations_list()
        self.N_TFMS = N_TFMS

    def apply(self, img, **params):
        transforms = A.Compose(list(np.random.choice(self.transform_list, self.N_TFMS, replace=False)))
        img = transforms(image=img)['image']
        return img

    def albumentations_list(self):
        geometric_transforms = A.OneOf([A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(0, 0.1), rotate_limit=10, p=1.0, interpolation=4),
                                        A.Rotate(limit=10, p=1.0, interpolation=4)
                                       ], p=0.5)

        lighting_transforms = A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                                       A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0),
                                       A.Equalize(p=1.0),
                                       A.RandomGamma(gamma_limit=(40, 140), p=1.0),
                                       A.RandomToneCurve(scale=0.4, p=1.0),
                                      ], p=0.5)

        distortion_transforms = A.OneOf([A.OpticalDistortion(distort_limit=0.4, p=1.0, interpolation=4),
                                         A.GridElasticDeform(num_grid_xy=(10, 10), magnitude=4, p=1.0, interpolation=4),
                                         A.GridDistortion(distort_limit=0.4, num_steps=5, p=1.0, interpolation=4),
                                         A.ElasticTransform(alpha=300, sigma=10, p=1.0, interpolation=4),
                                         A.Perspective(scale=(0.04, 0.06), keep_size=True, p=1.0, interpolation=4),
                                         A.ThinPlateSpline(scale_range=(0.04, 0.06), p=1.0, interpolation=4),
                                         A.RandomGridShuffle(grid=(2, 2), p=1.0)
                                        ], p=0.5)

        color_transforms = A.OneOf([A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1.0),
                                    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=0, val_shift_limit=0, p=1.0)
                                   ], p=0.5)

        noise_transforms = A.OneOf([A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
                                    A.GaussNoise(std_range=(0.02, 0.04), p=1.0),
                                    A.ShotNoise(scale_range=(0.01, 0.01), p=1.0)
                                   ], p=0.5)

        sharp_transforms = A.OneOf([A.Sharpen(alpha=(0.3, 0.5), lightness=(1.0, 1.0), p=1.0),
                                    A.Emboss(alpha=(0.4, 0.7), strength=(0.4, 0.7), p=1.0),
                                    A.UnsharpMask(blur_limit=(17, 21), alpha=1.0, threshold=1, p=1.0)
                                   ], p=0.5)

        transform_list = np.array([geometric_transforms,
                                   lighting_transforms,
                                   distortion_transforms,
                                   color_transforms,
                                   noise_transforms,
                                   sharp_transforms
                                   ], dtype='object')
        return transform_list

    def get_transform_init_args_names(self):
        return ("N_TFMS")

class MixUpTransform:
    def __init__(self,
                 num_classes,
                 p_mixup=0.10,
                 alpha=0.3,
                 device='cpu'):
        self.num_classes = num_classes
        self.p_mixup = p_mixup
        self.alpha = alpha
        self.device = device
        self.has_reached_max = False

    def transform(self, x, y):
        batch_size = x.size(0)
        p = np.random.rand(batch_size)
        mask = (p < self.p_mixup) & (p > 0.0)
        if (True in mask):
            indices = torch.randperm(batch_size)
            lam = torch.tensor(np.random.beta(self.alpha, self.alpha, size=batch_size), dtype=torch.float32).to(self.device)
            lam = torch.clamp(lam, 0.2, 0.8)
            lam = lam[:, None, None, None]
            lam[~mask] = 1
            x[mask] = x[mask] * lam[mask] + x[indices][mask] * (1 - lam[mask])
            y = torch.nn.functional.one_hot(y, self.num_classes).float().to(self.device)
            lam = lam[:, 0, 0, 0].unsqueeze(1)
            y[mask] = y[mask] * lam[mask] + y[indices][mask] * (1 - lam[mask])
        return x, y

class CutMixTransform:
    def __init__(self,
                 num_classes,
                 p_cutmix=0.10,
                 alpha=0.3,
                 device='cpu'):
        self.num_classes = num_classes
        self.p_cutmix = p_cutmix
        self.alpha = alpha
        self.device = device
        self.has_reached_max = False

    def transform(self, x, y):
        batch_size = x.size(0)
        p = torch.rand(batch_size)
        mask = (p < self.p_cutmix) & (p > 0.0)
        if True in mask:
            indices = torch.randperm(batch_size)
            lam = torch.tensor(np.random.beta(self.alpha, self.alpha, size=batch_size), dtype=torch.float32).to(self.device)
            lam = torch.clamp(lam, 0.2, 0.8)
            lam[~mask] = 1
            cut_ratios = torch.sqrt(1.0 - lam)  # Calculate the cut ratio.

            original_x = x.clone()
            # Generate random bounding box coordinates for the cut.
            for i in range(batch_size):
                if mask[i]:
                    bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), cut_ratios[i])
                    x[i, :, bbx1:bbx2, bby1:bby2] = original_x[indices[i], :, bbx1:bbx2, bby1:bby2]

            y = torch.nn.functional.one_hot(y, self.num_classes).float().to(self.device)
            lam = lam[:, None]
            y[mask] = y[mask] * lam[mask] + y[indices][mask] * (1 - lam[mask])
        return x, y

    def _rand_bbox(self, size, cut_rat):
        # Generate a random bounding box for CutMix.
        width = size[2]
        height = size[3]
        cut_w = (width * cut_rat).to(torch.int)
        cut_h = (height * cut_rat).to(torch.int)
        cx = torch.randint(0, width, size=cut_w.size(), dtype=cut_w.dtype)
        cy = torch.randint(0, height, size=cut_h.size(), dtype=cut_h.dtype)
        bbx1 = torch.clamp(cx - cut_w // 2, 0, width)
        bby1 = torch.clamp(cy - cut_h // 2, 0, height)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, width)
        bby2 = torch.clamp(cy + cut_h // 2, 0, height)
        return bbx1, bby1, bbx2, bby2

class GridMaskTransform:
    def __init__(self,
                 ratio,
                 unit_size_min,
                 unit_size_max,
                 random_offset,
                 p_gridmask):
        self.gridmask_transforms = A.GridDropout(ratio=ratio, unit_size_min=unit_size_min, unit_size_max=unit_size_max, random_offset=random_offset, p=p_gridmask)
        self.p_gridmask = p_gridmask

    def transform(self, x, y):
        x = x.permute(0, 2, 3, 1).numpy()
        for i in range(x.shape[0]):
            x[i] = self.gridmask_transforms(image=x[i])['image']
        x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        return x, y