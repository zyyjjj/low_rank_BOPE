from typing import List, Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

# outcome function

class Image(SyntheticTestFunction):
    r"""
    Class for generating rectangle images
    """
    dim = 4
    _bounds = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [0.,1.]])

    def __init__(self, num_pixels: int = 16):
        super().__init__()
        self.num_pixels = num_pixels
        self.pixel_size = 1 / self.num_pixels
        self.outcome_dim = num_pixels ** 2
    
    def evaluate_true(self, X):
        r"""
        Args:
            X: 4-dimensional input
        Returns:
            Y: (num_pixels^2)-dimensional array representing the images
        """

        # map real values in X to integer indices of pixels
        pixel_idcs = torch.div(X, self.pixel_size, rounding_mode="floor")

        Y = torch.zeros((*X.shape[:-1], self.num_pixels**2))

        for sample_idx in range(X.shape[-2]):

            row_start, col_start, row_end, col_end = pixel_idcs[sample_idx].numpy().astype(int)

            # swap if needed
            if row_start > row_end:
                row_start, row_end = row_end, row_start
            if col_start > col_end:
                col_start, col_end = col_end, col_start
            
            paint_it_black = [self.num_pixels * r + c \
                    for r in range(min(row_start, self.num_pixels-1), min(row_end+1, self.num_pixels)) \
                    for c in range(min(col_start, self.num_pixels-1), min(col_end+1, self.num_pixels))]
            
            Y[sample_idx, paint_it_black] = torch.ones(1, len(paint_it_black))

        return Y.to(torch.double)


# utility functions

class AreaUtil(torch.nn.Module):
    def __init__(self, binarize = True, weights: Optional[Tensor] = None):
        r"""
        Args:
            weights: `1 x outcome_dim` tensor 
        """
        super().__init__()
        self.weights = weights
        self.binarize = binarize
    
    def forward(self, Y: Tensor):
        if self.binarize:
            area = torch.sum((Y > 0.5).float(), dim =1)
        else:
            area = torch.sum(Y, dim = 1)

        if self.weights is not None:
            area = area * self.weights
        return area



class GradientAwareAreaUtil(torch.nn.Module):
    r"""
    Compute sum of pixel values penalized by the sum of difference between
    neighboring pixels pairs.
    """
    def __init__(self, penalty_param: float, image_shape: Tuple[float] = None):
        r"""
        
        """
        super().__init__()
        self.penalty_param = penalty_param
        self.image_shape = image_shape

    def compute_abs_gradient_sum(self, Y: Tensor):

        # TODO: check shape correctness

        nrows, ncols = self.image_shape

        res_all = []
        
        for y in Y:
            res = 0
            for row_idx in range(nrows):
                for col_idx in range(ncols):
                    i = row_idx * ncols + col_idx
                    if row_idx < nrows-1:
                        res += np.abs(y[i] - y[i+ncols])
                    if col_idx < ncols-1: 
                        res += np.abs(y[i] - y[i+1])
            res_all.append(res.item())

        return torch.tensor(res_all).unsqueeze(1)

    def forward(self, Y: Tensor):
        r"""
        
        """
        
        areas = torch.sum(Y, dim = 1).unsqueeze(1)
        abs_grad_sum = self.compute_abs_gradient_sum(Y)

        return areas - self.penalty_param * abs_grad_sum



# https://leetcode.com/problems/maximal-rectangle/solutions/264737/maximal-rectangle/ 
# https://leetcode.com/problems/largest-rectangle-in-histogram/editorial/ 
class LargestRectangleUtil(torch.nn.Module):
    r"""
    Compute the area of the largest rectangle in a binary image array, which
    is first rounded from a grayscale image array.
    """
    def __init__(self, image_shape: Tuple[float] = None):
        super().__init__()
        self.image_shape = image_shape
   
    # Get the maximum area in a histogram given its heights
    def maxRectangleHistogram(self, heights):
        stack = [-1]

        maxarea = 0
        for i in range(len(heights)):

            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                maxarea = max(maxarea, heights[stack.pop()] * (i - stack[-1] - 1))
            stack.append(i)

        while stack[-1] != -1:
            maxarea = max(maxarea, heights[stack.pop()] * (len(heights) - stack[-1] - 1))
        return maxarea


    def maximalRectangle(self, matrix: Tensor) -> int:

        if torch.sum(matrix).item() == 0:
            return 0

        maxarea = 0
        dp = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):

                # update the state of this row's histogram using the last row's histogram
                # by keeping track of the number of consecutive ones

                dp[j] = dp[j] + 1 if matrix[i][j].item() == 1 else 0

            # update maxarea with the maximum area from this row's histogram
            maxarea = max(maxarea, self.maxRectangleHistogram(dp))
        return maxarea
    
    def forward(self, Y: Tensor):
        Y_bin = (Y > 0.5).float() # num_samples x outcome_dim
        # then need to un-flatten outcome vector into image

        if self.image_shape is None:
            num_pixels = np.sqrt(Y.shape[-1])
            self.image_shape = (num_pixels, num_pixels)

        result = []
        for y_bin in Y_bin:
            result.append(self.maximalRectangle(torch.reshape(y_bin, self.image_shape)))
        
        return torch.tensor(result).unsqueeze(1) # TODO: check correctness


if __name__ == "__main__":

    NUM_PIXELS = 8

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
    print('Outcome data shape: ', Y.shape)

    # for y in Y:
    #     print(torch.reshape(y, (NUM_PIXELS, NUM_PIXELS)))

    # TEST Util func: vanilla sum of area    
    # print("computing areas")
    # area_util = AreaUtil()
    # print(area_util(Y))

    # TEST Util func: largest rectangle
    # print("computing largest rectangle for regular test cases")
    # max_rectangle_util = LargestRectangleUtil(image_shape = (NUM_PIXELS, NUM_PIXELS))
    # print(max_rectangle_util(Y))

    # TEST Util func: gradient penalized area
    print("Computing gradient penalized pixel sum")
    grad_pen_util = GradientAwareAreaUtil(penalty_param = 0.1, image_shape = (NUM_PIXELS, NUM_PIXELS))
    print(grad_pen_util(Y))