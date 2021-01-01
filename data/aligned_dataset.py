import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from imgaug import parameters as iap
from imgaug import augmenters as iaa
import numpy as np
import random
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        # define augmentations for doc data
        
        """Define our sequence of augmentation steps that will be applied to every image.""" 
        self.seq = self.get_aug()
      
        
        
    def get_aug(self):
        #sometimes_bg = lambda aug: iaa.Sometimes(0.3, aug)
        sometimes_contrast = lambda aug: iaa.Sometimes(0.3, aug)
        sometimes_noise = lambda aug: iaa.Sometimes(0.6, aug)
        sometimes_blur = lambda aug: iaa.Sometimes(0.6, aug)
        sometimes_degrade_quality = lambda aug: iaa.Sometimes(0.9, aug)
        sometimes_blend = lambda aug: iaa.Sometimes(0.2, aug)

        seq = iaa.Sequential(
                [
                # crop some of the images by 0-30% of their height/width
                # Execute 0 to 4 of the following (less important) augmenters per
                    # image. Don't execute all of them, as that would often be way too
                    # strong.
    #             iaa.SomeOf((0, 4),
    #                     [ 
                # change the background color of some of the images chosing any one technique
#                sometimes_bg(iaa.OneOf([
#                            iaa.AddToHueAndSaturation((-60, 60)),
#                            iaa.Multiply((0.6, 1), per_channel=True),
#                            ])),
                #change the contrast of the input images chosing any one technique    
                sometimes_contrast(iaa.OneOf([
                            iaa.LinearContrast((0.5,1.5)),
                            iaa.SigmoidContrast(gain=(3, 5), cutoff=(0.4, 0.6)),
                            iaa.CLAHE(tile_grid_size_px=(3, 21)),
                            iaa.GammaContrast((0.5,1.0))
                            ])),

                #add noise to the input images chosing any one technique 
                sometimes_noise(iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(3,8)),
                    iaa.CoarseDropout((0.001,0.01), size_percent=0.5),
                    iaa.AdditiveLaplaceNoise(scale=(3,10)),
                    iaa.CoarsePepper((0.001,0.01), size_percent=(0.5)),
                    iaa.AdditivePoissonNoise(lam=(3.0,10.0)),
                    iaa.Pepper((0.001,0.01)),
                    iaa.Snowflakes(),
                    iaa.Dropout(0.01,0.01),
                    ])),

                #add blurring techniques to the input image
                sometimes_blur(iaa.OneOf([
                    iaa.AverageBlur(k=(3)),
                    iaa.GaussianBlur(sigma=(1.0)),
                    ])),

                # add techniques to degrade the iamge quality
                sometimes_degrade_quality(iaa.OneOf([
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            iaa.Sharpen(alpha=(0.5), lightness=(0.75,1.5)),
                            iaa.BlendAlphaSimplexNoise(
                            foreground=iaa.Multiply(iap.Choice([1.5]), per_channel=False)
                            )
                            ])),

                # blend some patterns in the background    
                sometimes_blend(iaa.OneOf([
                            iaa.BlendAlpha(
                            factor=(0.6,0.8),
                            foreground=iaa.Sharpen(1.0, lightness=1),

                            background=iaa.CoarseDropout(p=0.1, size_px=np.random.randint(30))),

                            iaa.BlendAlphaFrequencyNoise(exponent=(-4),
                                       foreground=iaa.Multiply(iap.Choice([0.5]), per_channel=False)
                                       ),
                            iaa.BlendAlphaSimplexNoise(
                            foreground=iaa.Multiply(iap.Choice([0.5]), per_channel=True)
                            )
                      ])), 

                    ])
        return seq

        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
 
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        
        #print(self.opt.isTrain)
        if self.opt.isTrain:
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),aug=True,seg=self.seq)
        else:
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),aug=True,seg=self.seq)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1),aug=False,seg=self.seq)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
