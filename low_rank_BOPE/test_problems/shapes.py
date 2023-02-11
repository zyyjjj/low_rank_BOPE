import torch
from torch import Tensor

# outcome function

class Image():
    r"""
    Class for generating rectangle images
    """
    def __init__(self, num_pixels: int = 16):
        self.num_pixels = num_pixels
        self.pixel_size = 1 / self.num_pixels
    
    def forward(self, X):
        r"""
        Args:
            X: 4-dimensional input
        Returns:
            Y: 256-dimensional array representing 16x16 images
        """

        # map real values in X to integer indices of pixels
        pixel_idcs = torch.div(X, 1/self.pixel_size, rounding_mode="floor")

        # pixel_idx[..., 0:2] specify the upper left corner
        # pixel_idx[..., 2:4] specify the lower right corner
        # resulting image 

        # TODO: there must be a better way to handle this in batch

        Y = torch.zeros((X.shape[:-1], self.num_pixels**2))

        for sample_idx in range(X.shape[0]): # TODO: could be shape erros

            row_start, col_start, row_end, col_end = pixel_idcs[sample_idx]

            paint_it_black = [self.num_pixels * r + c \
                    for r in range(row_start, row_end+1) \
                    for c in range(col_start, col_end+1)]
            
            Y[sample_idx, paint_it_black] = torch.ones(1, len(paint_it_black))

        return Y


# utility function -- TODO: make into class

def area(image):
    pass

def weighted_area(image, weights):
    pass