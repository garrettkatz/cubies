import itertools as it
import numpy as np

# grounded or wildcard
class PatternDatabase:

    def __init__(self, patterns, macros, domain):
        # macros[i] is action sequence for patterns[i]
        # 0 in pattern matches anything, 1 <= v <= 6 matches specific color
        self.match_index = 0
        self.matched = False

        # save originals
        self.orig_patterns = patterns
        self.orig_macros = macros

        # expand db with rotational symmetries
        self.patterns = np.empty((24*len(patterns), domain.state_size()), dtype=int)
        self.syms = []
        self.macros = []
        for p, pattern in enumerate(patterns):
            for s, sym_pattern in enumerate(domain.symmetries_of(pattern)):
                self.patterns[24*p + s] = sym_pattern
                self.syms.append(domain.inverse_symmetry_of(s))
                self.macros.append(macros[p])

        # trace query history
        self.match_counts = np.zeros(len(patterns), dtype=int) # aggregates across symmetry
        self.hit_counts = np.zeros(self.patterns.shape, dtype=int)
        self.num_queries = 0

    def query(self, state):

        # brute query
        hits = (self.patterns == state) | (self.patterns == 0)
        self.match_index = np.flatnonzero(hits.all(axis=1))

        # # progressive query
        # matches = np.arange(self.patterns.shape[0])
        # for k in range(len(state)):
        #     matches = matches[(self.patterns[matches, k] == state[k]) | (self.patterns[matches, k] == 0)]
        # self.match_index = matches

        # update trace
        self.match_counts[self.match_index // 24] += 1
        self.hit_counts += hits
        self.num_queries += 1

        # return status
        self.matched = self.match_index.size > 0
        return self.matched

    def result(self): # assumes match was not False
        m = self.match_index[0] # what about multiple matches?
        return self.syms[m], list(self.macros[m])


# # variables and unification
# class PatternDatabase:

#     def __init__(self, patterns, macros):
#         # macros[i] is action sequence for patterns[i]
#         # 0 in pattern matches anything, 1 <= v <= 6 matches specific color, v > 6 is unbound variable
#         # all occurrences of variable must match same color in state
#         self.patterns = patterns
#         self.macros = macros
#         self.match_index = 0
#         self.matched = False

#         # bindings[i,v] is value bound to variable v in pattern i
#         self.bindings = np.tile(np.arange(patterns.max()+1), (patterns.shape[0], 1))

#     def query(self, state):

#         # copy initial patterns and bindings for in-place variable grounding
#         patterns = self.patterns.copy()
#         bindings = self.bindings.copy()

#         # initialize index of leading patterns that unify with state (will be pruned)
#         unified = np.arange(self.patterns.shape[0])

#         # iterate over state values to progressively bind pattern variables and unify
#         for k in range(patterns.shape[1]):

#             # collect indices of leading patterns that still unify with state and have variables at entry k
#             var_index = unified[patterns[unified,k] > 6] # variables are numbers > 6
#             if var_index.size > 0:

#                 # substitute values for variables that were already bound in a previous iteration
#                 patterns[var_index, k:k+1] = np.take_along_axis(bindings[var_index,:], patterns[var_index, k:k+1], axis=1)

#                 # collect remaining indices of patterns that still have  unbound variables at entry k
#                 var_index = var_index[patterns[var_index,k] > 6]

#             # bind and ground remaining variables to value in state at entry k
#             if var_index.size > 0:

#                 # update bindings
#                 np.put_along_axis(bindings[var_index,:], patterns[var_index, k:k+1], state[k], axis=1)

#                 # ground patterns
#                 patterns[var_index, k] = state[k]

#             # discard any patterns that no longer unify (0 unifies with everything)
#             unified = unified[(patterns[unified,k] == state[k]) | (patterns[unified,k] == 0)]
#             if unified.size == 0: break #  if no patterns unify anymore, stop early

#         # cache query result
#         self.match_index = unified # what about multiple matches?
#         self.matched = unified.size > 0

#         return self.matched

#     def result(self): # assumes match was not False
#         return list(self.macros[self.match_index[0]])

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
    
    # ### test pdb with variables
    # patterns = np.array([
    #     [1,2,3,4,5,6],
    #     [1,2,3,4,5,0],
    #     [0,2,3,7,5,7],
    #     [0,2,8,7,8,7],
    #     ])
    # macros = [(1,),(2,),(3,),(4,)]

    # pattern_database = PatternDatabase(patterns, macros)

    # states = [
    #     np.array([1,2,3,4,5,6]),
    #     np.array([1,2,3,4,5,5]),
    #     np.array([1,2,3,4,5,4]),
    #     np.array([1,2,3,4,3,4]),
    #     np.array([2,3,4,5,6,1]),
    #     ]

    # for state in states:

    #     matched = pattern_database.query(state)
    #     print(state)
    #     print(matched)
    #     if matched:
    #         print(pattern_database.patterns[pattern_database.match_index[0]])
    #         print(pattern_database.result())

    # print()
    # assert pattern_database.query(states[0])
    # assert pattern_database.result() == [1]
    # assert pattern_database.query(states[1])
    # assert pattern_database.result() == [2]
    # assert pattern_database.query(states[2])
    # assert pattern_database.result() == [2]
    # assert pattern_database.query(states[3])
    # assert pattern_database.result() == [4]
    # assert pattern_database.query(states[4]) == False

    # ### test grounded pdb with wildcards

    from cube import CubeDomain
    domain = CubeDomain(2)
    state = domain.solved_state()
    patterns = np.array([
        state,
        domain.perform((1,0,1), state),
        state,
    ])
    patterns[2, (patterns[0] != patterns[1])] = 0
    macros = [(0,), (1,), (2,)]
    
    pattern_database = PatternDatabase(patterns, macros, domain)
    states = [
        state,
        domain.perform((1,0,1), state),
        domain.perform((1,0,2), state),
        domain.symmetries_of(state)[domain.inverse_symmetry_of(15)],
        domain.perform((2,0,1), state),
    ]

    for state in states:

        matched = pattern_database.query(state)
        print(state)
        print(matched)
        if matched:
            print(pattern_database.match_index[0])
            print(pattern_database.patterns[pattern_database.match_index[0]])
            print(pattern_database.result())

    print()
    assert pattern_database.query(states[0])
    assert pattern_database.result() == (20, [0])
    assert pattern_database.query(states[1])
    assert pattern_database.result() == (20, [1])
    assert pattern_database.query(states[2])
    assert pattern_database.result() == (20, [2])
    assert pattern_database.query(states[3])
    assert pattern_database.result() == (15, [0])
    assert pattern_database.query(states[4]) == False

