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

        # bindings[i,v] is value bound to variable v in pattern i
        self.bindings = np.tile(np.arange(patterns.max()+1), (patterns.shape[0], 1))

    def query(self, state):

        # copy initial patterns and bindings for in-place variable grounding
        patterns = self.patterns.copy()
        bindings = self.bindings.copy()

        # initialize index of leading patterns that unify with state (will be pruned)
        unified = np.arange(self.patterns.shape[0])

        # iterate over state values to progressively bind pattern variables and unify
        for k in range(patterns.shape[1]):

            # collect indices of leading patterns that still unify with state and have variables at entry k
            var_index = unified[patterns[unified,k] > 6] # variables are numbers > 6
            if var_index.size > 0:

                # substitute values for variables that were already bound in a previous iteration
                patterns[var_index, k:k+1] = np.take_along_axis(bindings[var_index,:], patterns[var_index, k:k+1], axis=1)

                # collect remaining indices of patterns that still have  unbound variables at entry k
                var_index = var_index[patterns[var_index,k] > 6]

            # bind and ground remaining variables to value in state at entry k
            if var_index.size > 0:

                # update bindings
                np.put_along_axis(bindings[var_index,:], patterns[var_index, k:k+1], state[k], axis=1)

                # ground patterns
                patterns[var_index, k] = state[k]

            # discard any patterns that no longer unify (0 unifies with everything)
            unified = unified[(patterns[unified,k] == state[k]) | (patterns[unified,k] == 0)]
            if unified.size == 0: break #  if no patterns unify anymore, stop early

        # cache query result
        self.match_index = unified # what about multiple matches?
        self.matched = unified.size > 0

        return self.matched

    def result(self): # assumes match was not False
        return list(self.macros[self.match_index[0]])

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
        print(state)
        print(matched)
        if matched:
            print(pattern_database.patterns[pattern_database.match_index[0]])
            print(pattern_database.result())

    print()
    assert pattern_database.query(states[0])
    assert pattern_database.result() == [1]
    assert pattern_database.query(states[1])
    assert pattern_database.result() == [2]
    assert pattern_database.query(states[2])
    assert pattern_database.result() == [2]
    assert pattern_database.query(states[3])
    assert pattern_database.result() == [4]
    assert pattern_database.query(states[4]) == False



