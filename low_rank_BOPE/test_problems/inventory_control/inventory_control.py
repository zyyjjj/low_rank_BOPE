import numpy as np
import torch
from torch import Tensor
from botorch.test_functions import SyntheticTestFunction


PARAMS = {
    'demand_mean' : 5, 
    'demand_std' : 3,
    'lead_time' : 10, # tau in the slides
    'stockout_penalty' : 0.1, # p in the slides
    'holding_cost' : 0.01, # h in the slides
    'K' : 1, 
    'c' : 0.1 # order cost parameters
}

# code credit to Peter
def step(state, Q, R, params):
    r"""
    Run one day forward and return the updated state.

    Args:
        state: a dictionary tracking 'inventory', 'days_left', 'cost'
        Q: order quantity
        R: reorder level
        params: a dictionary of parameters
    """

    # Check whether an order has arrived
    if state['days_left'] == 0:
        # If it has, unload the truck and add the contents to our inventory
        state['inventory'] += Q
        # Mark the state to say that a new order hasn't been placed
        state['days_left'] = -1
    # Otherwise, if an order is in transit, it gets one day closer
    elif state['days_left']>0:
        state['days_left'] -= 1
    
    # Simulate the demand on this day, and fill it from our inventory.
    # This may make our inventory negative.
    # Also calculate stockout costs
    demand = np.random.normal(params['demand_mean'], params['demand_std'])
    
    # Calculate stockout costs
    onhand = np.max([0,state['inventory']])
    backordered = np.max([0,demand - onhand])
    state['cost']+= params['stockout_penalty'] * backordered
    
    # Fill demand from inventory, backordering if needed
    state['inventory'] -= demand
    
    # Pay holding cost
    state['cost'] += params['holding_cost']*np.max([state['inventory'],0])
    
    # If our inventory is below the reorder level, and an order is 
    # not on its way, then place an order
    if state['inventory'] < R and state['days_left'] == -1:
        state['days_left'] = params['lead_time']
        state['cost'] += params['K'] + params['c']*Q
        
    return state


class Inventory(SyntheticTestFunction):
    r"""
    Simulate an inventory control problem using the QR policy.
    """
    dim = 2
    _bounds = torch.tensor([[0., 1.], [0., 1.]])

    def __init__(
            self,
            duration: int,
            init_inventory: int, # TODO: do we want to keep this fixed?
            params = PARAMS
    ):
        super().__init__()
        self.duration = duration
        self.init_inventory = init_inventory
        self.params = params

    def evaluate_true(self, X: Tensor):
        r"""
        Args:
            X: `num_samples x 2` tensor, where first column is Q and second is R
        """
        Y = []
        for x in X:
            Y.append(self.evaluate_true_single(x))
        
        return torch.vstack(Y)

    def evaluate_true_single(self, x: Tensor):
        r"""
        Simulate an inventory trajectory for one policy.
        Args:
            x: `1x2` tensor, where first entry is Q and second is R # TODO: confirm shape
        """

        x_np = x.detach.numpy()
        state = {
            'inventory' : self.init_inventory, 
            'days_left' : -1, 
            'cost' : 0
        }
        inventory = []
        days_left = []
        cost = []

        for t in range(self.duration):
            inventory.append(state['inventory'])
            days_left.append(state['days_left'])
            cost.append(state['cost'])
            state = step(state, Q=x_np[0], R=x_np[1], params=self.params)

        return torch.tensor(inventory)
        


class InventoryUtil(torch.nn.Module):
    r"""Do something"""
    def __init__(
            self,
            # more
    ):
        super().__init__()
        pass

    def forward(self, Y: Tensor):
        r"""
        Args:
            Y: 
        """
        pass