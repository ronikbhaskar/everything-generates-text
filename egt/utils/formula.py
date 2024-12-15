

from copy import deepcopy

class MCNF:
    """
    Conjunctive Normal Form is a standard Boolean formula structure.
    Many-Valued Conjunctive Normal Forms have been done before.
    In this case, I came up with a Many-Valued CNF that allows for a gradient of truth values,
    allowing for better thresholding in decision trees (I hope).
    
    As a lower bound, it is at least as expressive as standard CNF.

    Given a number of clauses n, and clause sets S_k for k in [n], 
    we evaluate a sequential, categorical feature as:
    1/n * sum_{k=1}^n 1_{f_k in S_k}

    In this case, AND is replaced with the arithmetic mean, 
    and each variable in the disjunction clauses is v_{i,k} := f_k == S_k[i]
    """

    def __init__(
        self,
        num_features: int,
    ):
        assert num_features > 0
        self.num_features = num_features
        self.clause_sets = [set() for _ in range(num_features)]

    def update(self, value, feature_idx):
        """
        returns new MCNF
        does NOT update in place
        """

        mcnf = MCNF(self.num_features)
        mcnf.clause_sets = deepcopy(self.clause_sets)
        mcnf.clause_sets[feature_idx].add(value)

        return mcnf
    
    def evaluate(self, sample):
        return 1 / self.num_features * sum((
            int(sample[i] in clause_set)
            for i, clause_set in enumerate(self.clause_sets)))
