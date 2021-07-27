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

    from cube import CubeDomain
    domain = CubeDomain(2)
    state = domain.solved_state()
    
    from tree import SearchTree
    tree = SearchTree(domain, max_depth=1)
    patterns = np.array([state[permutation] for _, permutation in tree])
    patterns[:,5] = 0
    patterns[:,6] = 8
    macros = [actions for actions, _ in tree]
    
    # patterns = state[np.newaxis,:].copy()
    # patterns[:,0] = 2
    # patterns[:,1] = 0
    # macros = [((1,0,1),)]

    pattern_database = PatternDatabase(patterns, macros)

    # Non-empty macro if state matches a database pattern
    matched = pattern_database.query(state)
    print(matched)
    print(pattern_database.result())

    matched = pattern_database.query(domain.perform((1,0,1), state))
    print(matched)
    print(pattern_database.result())


