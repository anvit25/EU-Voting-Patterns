import numpy as np
import scipy.linalg as la
import pandas as pd
import os

class Data:
    def __init__(self, type = 'featured'):
        if type not in ['featured', 'main']:
            raise ValueError("Invalid vote type. Please choose 'featured' or 'main'.")
        data_folder = os.path.join(f'{type}_data')

        member_country_path = os.path.join(data_folder, 'member_country.csv')
        self.member_country = pd.read_csv(member_country_path, index_col="member_id")
        self.member_country = self.member_country.astype(int)

        self.country_names = list(self.member_country.columns.copy())
        self.country_sizes = self.member_country.sum().to_numpy() #TO FIX
        # self.country_sizes = np.array([]) # Ben will fix this

        member_vote_for_path = os.path.join(data_folder, 'member_vote_for.csv')
        self.member_vote_for = pd.read_csv(member_vote_for_path, index_col="member_id")
        assert np.all(self.member_vote_for.index == self.member_country.index)

        member_vote_against_path = os.path.join(data_folder, 'member_vote_against.csv')
        self.member_vote_against = pd.read_csv(member_vote_against_path, index_col="member_id")
        assert np.all(self.member_vote_against.index == self.member_country.index)


    def get_member_country(self, pandas = False):
        if pandas:
            return self.member_country
        return self.member_country.to_numpy()
    
    def get_member_vote(self, pandas = False, yes_votes = True):
        if pandas:
            if yes_votes:
                return self.member_vote_for
            return self.member_vote_against
        if yes_votes:
            return self.member_vote_for.to_numpy()
        return self.member_vote_against.to_numpy()
    
    def get_vote_country(self, yes_votes = True, normalize = False):
        mem_coun = self.get_member_country() 
        if normalize:
            mem_coun = mem_coun/self.country_sizes
        if yes_votes:
            return self.get_member_vote().T @ mem_coun
        return self.get_member_vote(yes_votes = False).T @ mem_coun
    
    def get_country_country(self, yes_votes = True, normalize = False):
        vote_coun = self.get_vote_country(yes_votes = yes_votes, normalize = normalize)
        return vote_coun.T @ vote_coun


class BaseCommunityDetection(Data):
    def __init__(self, dtype = 'featured'):
        self.name = "Base"
        super().__init__(dtype)
    
    def __str__(self):
        return self.name
    
    def fit(self):
        raise NotImplementedError("Please implement the fit method. Do not remove the following 3 lines.")
        return self
    
    def matrix_power(self, A, n):
        """Compute the nth power of a positive definite matrix."""
        vals, vecs = la.eig(A)
        return vecs @ np.diag(vals**n) @ vecs.T

    def sorted_eigh(self, A):
        """Compute the eigenvalues and eigenvectors of a matrix, sorted by decreasing eigenvalues."""
        vals, vecs = la.eigh(A)
        idx = np.argsort(vals)[::-1]
        return vals[idx], vecs[:,idx]
    
class TemplateMethod(BaseCommunityDetection):
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
    def __init__(self, dtype = 'featured'):
        self.name = "Template"
        super().__init__(dtype)
    
    def fit(self):
        '''
        Saves a pd.DataFrame in the attribute called self.labels with the following columns:
        - "Name"  - Country name
        - "Label" - Cluster number
        '''
        self.labels = pd.DataFrame({
            "Name": self.country_names,
            "Label": np.random.randint(0, 3, len(self.country_names))
        })
        return self

if __name__ == "__main__":
    A = np.random.uniform(0, 1, (4, 2))
    A = A.T @ A
    B = la.fractional_matrix_power(A, 1/3)
    assert np.allclose(BaseCommunityDetection.matrix_power(A, 1/3), B)

    data = Data('featured')