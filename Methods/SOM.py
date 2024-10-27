import numpy as np
import pandas as pd
from tqdm import trange
from .utils import BaseCommunityDetection

class SOM(BaseCommunityDetection):
    '''
        Template class for community detection methods.
        The dtype parameter can be either 'featured' or 'main'.
        It is used to determine which data to use.

        All parameters of the model that you would want to tune should be passed in the constructor (i.e. __init__).

        Some useful methods include:
        - self.get_member_country() -> np.ndarray
        - self.get_member_vote() -> np.ndarray
        - self.get_vote_country() -> np.ndarray
        - self.get_country_country() -> np.ndarray
        - self.sorted_eigh(A) -> Tuple[np.ndarray, np.ndarray]
        - self.matrix_power(A, n) -> np.ndarray (Well defined only for positive definite matrices)

        All these methods have some optional parameters. Check the Data class for more information.
    '''
    def __init__(self, grid_size = 20, locality = 1/2, dtype = 'featured'):
        self.name = "Self Organizing Map"
        self.grid_size = grid_size
        self.locality = locality
        super().__init__(dtype)
        self.data = np.log(1/2 + self.get_vote_country(normalize=True))
        num_votes = self.data.shape[0]
        self.grid = np.random.uniform(0, 1, (grid_size, grid_size, num_votes))

    def calculate_weights(self, x, y):
        grid_indices = np.indices((self.grid_size, self.grid_size)) - np.array([x, y])[:, None, None]
        return np.exp(-np.sum(grid_indices**2, axis=0)*self.locality)

    def one_step(self):
        delta_grid = np.zeros_like(self.grid)
        for j in range(self.data.shape[1]):
            distances = np.sum((self.grid - self.data[:, j])**2, axis=2)
            x,y = np.unravel_index(np.argmin(distances), (self.grid_size, self.grid_size))
            weight = 0.1 * self.calculate_weights(x, y)
            delta_grid += weight[:, :, None] * (self.data[:, j] - self.grid)
        self.grid += delta_grid
        return np.linalg.norm(delta_grid)
    
    def fit(self, n_iter = 1000):
        '''
        Saves a pd.DataFrame in the attribute called self.labels with the following columns:
        - "Name"  - Country name
        - "Label" - Cluster number
        '''
        norm_delta = 1
        for _ in (pbar := trange(n_iter)):
            if norm_delta < 1e-5:
                break
            norm_delta = self.one_step()
            pbar.set_description(f"Norm Delta: {norm_delta}")

        self.embedding = np.zeros((self.data.shape[1], 2))
        for i in range(self.data.shape[1]):
            x, y = np.unravel_index(np.argmin(np.sum((self.grid - self.data[:, i])**2, axis=2)), (self.grid_size, self.grid_size))
            self.embedding[i] = [x, y]

        self.labels = pd.DataFrame({
            "Name": self.country_names,
            "Label": self.generate_labels(3)
        })
        return self
    
    def visualize_weights(self):
        x = y = int(self.grid_size/2)
        return self.calculate_weights(x, y)
