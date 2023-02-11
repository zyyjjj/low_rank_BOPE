from typing import Optional

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

# outcome function

def swap(a,b):
    tmp = a
    a=b
    b=tmp

    return a, b

class Image(SyntheticTestFunction):
    r"""
    Class for generating rectangle images
    """
    dim = 4
    _bounds = torch.tensor([[0, 1], [0, 1], [0, 1], [0,1]])

    def __init__(self, num_pixels: int = 16):
        super().__init__()
        self.num_pixels = num_pixels
        self.pixel_size = 1 / self.num_pixels
    
    def evaluate_true(self, X):
        r"""
        Args:
            X: 4-dimensional input
        Returns:
            Y: 256-dimensional array representing 16x16 images
        """

        # map real values in X to integer indices of pixels
        pixel_idcs = torch.div(X, self.pixel_size, rounding_mode="floor")

        Y = torch.zeros((*X.shape[:-1], self.num_pixels**2))

        for sample_idx in range(X.shape[-2]): # TODO: there could be shape erros

            row_start, col_start, row_end, col_end = pixel_idcs[sample_idx].numpy().astype(int)

            # swap if needed
            if row_start > row_end:
                row_start, row_end = swap(row_start, row_end)
            if col_start > col_end:
                col_start, col_end = swap(col_start, col_end)
            
            paint_it_black = [self.num_pixels * r + c \
                    for r in range(min(row_start, self.num_pixels-1), min(row_end+1, self.num_pixels)) \
                    for c in range(min(col_start, self.num_pixels-1), min(col_end+1, self.num_pixels))]
            
            Y[sample_idx, paint_it_black] = torch.ones(1, len(paint_it_black))

        return Y


# utility function

class AreaUtil(torch.nn.Module):
    def __init__(self, weights: Optional[Tensor] = None):
        r"""
        Args:
            weights: `1 x outcome_dim` tensor 
        """
        super().__init__()
        self.weights = weights
    
    def forward(self, Y: Tensor):
        area = torch.sum(Y, dim = 1)
        if self.weights is not None:
            area = area * self.weights
        return area




if __name__ == "__main__":

    NUM_PIXELS = 16

    image_problem = Image(num_pixels = NUM_PIXELS)
    test_X = torch.tensor(
        [
            # edge cases: largest area possible (all pixels black)
            [0,1,1,0],
            [1,0,0,1],
            [1,1,0,0],
            [0,0,1,1],
            # edge cases: smallest area possible (pixel on a corner)
            [0,0,0,0],
            [1,0,1,0],
            [0,1,0,1],
            [1,1,1,1],
            # generic cases
            [0.1, 0.1, 0.6, 0.6],
            [0.1, 0.8, 0.1, 0.6],
            [0.6, 0.1, 0.1, 0.8],
            [0.1, 0.6, 0.8, 0.1]
        ])

    Y = image_problem(test_X)
    for y in Y:
        print(torch.reshape(y, (NUM_PIXELS, NUM_PIXELS)))
    
    print("computing areas")
    area_util = AreaUtil()
    print(area_util(Y))
