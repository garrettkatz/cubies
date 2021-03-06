import itertools as it
import numpy as np

def pdb_query(state, patterns, wildcards):

    # brute
    if len(patterns) < 10000:
        index = np.flatnonzero(((state[3:] == patterns[:,3:]) | wildcards[:,3:]).all(axis=1)) # first three facies invariant

    # progressive
    else:
        index = np.flatnonzero((state[3] == patterns[:,3]) | wildcards[:,3]) # first three facies invariant
        for k in range(4, patterns.shape[1]):
            if len(index) == 0: return index
            index = index[(state[k] == patterns[index, k]) | wildcards[index, k]]

    return index

# grounded or wildcard
class PatternDatabase:

    def __init__(self, patterns, wildcard, macros, domain, orientation_neutral=False):
        # macros[i] is action sequence for patterns[i]
        # patterns[i,j]: 1 <= v <= 6 matches specific color in state
        # patterns[i,j] need not match if wildcard[i,j] == 1

        self.orientation_neutral = orientation_neutral

        if orientation_neutral:
            # expand db with rotational symmetries
            self.patterns = np.empty((24*len(patterns), domain.state_size()), dtype=int)
            self.wildcard = np.empty((24*len(patterns), domain.state_size()), dtype=bool)
            self.macros = []
            self.syms = []
            for p in range(len(patterns)):
                sym_patterns = domain.orientations_of(patterns[p])
                sym_wildcard = domain.orientations_of(wildcard[p])
                for s in range(24):
                    self.patterns[24*p + s] = sym_patterns[s]
                    self.wildcard[24*p + s] = sym_wildcard[s]
                    self.macros.append(macros[p])
                    self.syms.append(domain.inverse_symmetry_of(s))
        else:
            self.patterns = np.array(patterns)
            self.wildcard = np.array(wildcard)
            self.macros = list(macros)
            self.syms = [0] * len(patterns)

        # initialize matches and traces
        self.reset()

    def reset(self):

        # match results
        self.match_mask = np.zeros(len(self.patterns), dtype=bool)
        self.match_index = 0
        self.matched = False

        # trace query history
        sym_mod = 24 if self.orientation_neutral else 1
        self.match_counts = np.zeros(len(self.patterns) // sym_mod, dtype=int) # aggregates across symmetry
        self.miss_counts = np.zeros(self.patterns.shape, dtype=int)
        self.num_queries = 0

    def query(self, state):
        index = pdb_query(state, self.patterns, self.wildcard)

        # self.match_mask = np.zeros(len(patterns), dtype=bool)
        # self.match_mask[index] = True
        self.match_index = tuple(index[:1])
        self.matched = len(index) > 0
        # if self.matched:
        #     print("matched index:")
        #     print(index)
        #     print(self.match_index)
        return self.matched

        # # fast brute
        # self.match_mask = ((self.patterns == state) | (self.wildcard)).all(axis=1)
        # self.match_index = (self.match_mask.argmax(),)
        # self.matched = self.match_mask[self.match_index[0]]
        # return self.matched

        # # brute query
        # hits = (self.patterns == state) | (self.wildcard)
        # self.match_index = np.flatnonzero(hits.all(axis=1))

        # # # progressive query
        # # matches = np.arange(self.patterns.shape[0])
        # # for k in range(len(state)):
        # #     matches = matches[(self.patterns[matches, k] == state[k]) | (self.patterns[matches, k] == 0)]
        # # self.match_index = matches

        # # # update trace
        # # if self.match_index.size > 0:
        # #     idx = self.match_index[0]
        # #     sym_mod = 24 if self.orientation_neutral else 1
        # #     self.match_counts[idx // sym_mod] += 1
        # #     self.miss_counts[:idx] += ~hits[:idx]
        # # else:
        # #     self.miss_counts += ~hits
        # # self.num_queries += 1

        # # return status
        # self.matched = self.match_index.size > 0
        # return self.matched

    def result(self): # assumes match was not False
        m = self.match_index[0] # what about multiple matches?
        return m
        # return self.syms[m], tuple(self.macros[m])


if __name__ == "__main__":

    # ### test grounded pdb with wildcards

    from cube import CubeDomain
    domain = CubeDomain(2)
    state = domain.solved_state()
    patterns = np.array([
        state,
        domain.perform((1,0,1), state),
        state,
    ])
    # patterns[2, (patterns[0] != patterns[1])] = 0
    wildcard = np.zeros(patterns.shape, dtype=bool)
    wildcard[2, (patterns[0] != patterns[1])] = True
    macros = [(0,), (1,), (2,)]
    
    pattern_database = PatternDatabase(patterns, wildcard, macros, domain)
    states = [
        state,
        domain.perform((1,0,1), state),
        domain.perform((1,0,2), state),
        domain.orientations_of(state)[domain.inverse_symmetry_of(15)],
        domain.perform((2,0,1), state),
        domain.orientations_of(domain.perform((1,0,2), state))[domain.inverse_symmetry_of(15)],
    ]

    for state in states:

        matched = pattern_database.query(state)
        print(state)
        print(matched)
        if matched:
            idx = pattern_database.match_index[0]
            print(idx)
            print(pattern_database.patterns[idx] * (1 - pattern_database.wildcard[idx]))
            print(pattern_database.result())

    print()
    assert pattern_database.query(states[0])
    assert pattern_database.result() == (0, [0])
    assert pattern_database.query(states[1])
    assert pattern_database.result() == (0, [1])
    assert pattern_database.query(states[2])
    assert pattern_database.result() == (0, [2])
    assert pattern_database.query(states[3])
    assert pattern_database.result() == (15, [0])
    assert pattern_database.query(states[4]) == False
    assert pattern_database.query(states[5])
    assert pattern_database.result() == (15, [2])

