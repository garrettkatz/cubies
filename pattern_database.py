import numpy as np

class PatternDatabase:

    def __init__(self, patterns, macros):
        # macros[i] is action sequence for patterns[i]
        # 0 in pattern matches anything, 1 <= v <= 6 matches specific color, v > 6 is unbound variable
        # all occurrences of variable must match same color in state
        self.patterns = patterns
        self.macros = macros
        self.match_index = 0
        self.matched = False

    def query(self, state):

        # copy patterns for variable binding
        patterns = self.patterns.copy()

        # initialize mask for successful unifications
        unified = np.ones(self.patterns.shape[0], dtype=bool)

        # unify state with patterns
        for k in range(state.size):

            # select unbound variables in unified leading patterns
            unbound = unified & (patterns[:,k] > 6)
            if unbound.any():
                patterns[unbound,k:] = np.where(patterns[unbound,k:] == patterns[unbound,k:k+1], state[k], patterns[unbound,k:])

            # updated mask of patterns that still unify with state
            unified &= (patterns[:,k] == state[k]) | (patterns[:,k] == 0)
        
        self.match_index = unified.argmax() # what about multiple matches?
        self.matched = (unified[self.match_index] != False)

        return self.matched

    def result(self): # assumes match was not False
        return list(self.macros[self.match_index])

if __name__ == "__main__":

    # from cube import CubeDomain
    # from tree import SearchTree
    # domain = CubeDomain(2)
    # state = domain.solved_state()    
    # tree = SearchTree(domain, max_depth=1)
    # patterns = np.array([state[permutation] for _, permutation in tree])
    # patterns[:,5] = 0
    # patterns[:,6] = 8
    # macros = [actions for actions, _ in tree]
    
    # patterns = state[np.newaxis,:].copy()
    # patterns[:,0] = 2
    # patterns[:,1] = 0
    # macros = [((1,0,1),)]
    
    # states = [domain.perform((1,0,1), state)]
    
    patterns = np.array([
        [1,2,3,4,5,6],
        [1,2,3,4,5,0],
        [0,2,3,7,5,7],
        [0,2,8,7,8,7],
        ])
    macros = [(1,),(2,),(3,),(4,)]

    pattern_database = PatternDatabase(patterns, macros)
    
    states = [
        np.array([1,2,3,4,5,6]),
        np.array([1,2,3,4,5,5]),
        np.array([1,2,3,4,5,4]),
        np.array([1,2,3,4,3,4]),
        np.array([2,3,4,5,6,1]),
        ]

    for state in states:

        matched = pattern_database.query(state)
        print()
        print(state)
        print(matched)
        if matched:
            print(pattern_database.patterns[pattern_database.match_index])
            print(pattern_database.result())



