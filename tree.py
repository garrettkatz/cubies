"""
Memoize a depth-limited, BFS search tree for the cube domain
Paths and resulting permutations to each node are precomputed
"""
import numpy as np

class SearchTree:

    def __init__(self, domain, max_depth, orientation_neutral=True):

        permutation = np.arange(domain.state_size())
        actions = tuple()
        
        explored = set([permutation.tobytes()])
        layers = {0: [(actions, permutation)]}
        
        for depth in range(max_depth):
            layers[depth+1] = []

            for actions, permutation in layers[depth]:
                for action in domain.valid_actions(permutation):

                    # get child state permutation
                    new_actions = actions + (action,)
                    new_permutation = domain.perform(action, permutation)
                    
                    # skip if already explored
                    if orientation_neutral:
                        orientations = domain.orientations_of(new_permutation)
                        if any([perm.tobytes() in explored for perm in orientations]): continue
                    else:
                        if new_permutation.tobytes() in explored: continue

                    # otherwise add the new state to the frontier and explored set
                    explored.add(new_permutation.tobytes())
                    layers[depth+1].append((new_actions, new_permutation))
        
        self._layers = layers
    
    def __iter__(self):
        for depth in range(len(self._layers)):
            for actions, permutation in self._layers[depth]:
                yield actions, permutation

    def rooted_at(self, state, up_to_depth=None):
        if up_to_depth is None: up_to_depth = len(self._layers)-1
        for depth in range(up_to_depth+1):
            for actions, permutation in self._layers[depth]:
                yield actions, state[permutation]


if __name__ == "__main__":
    
    # from cube import CubeDomain
    # domain = CubeDomain(3)
    # A = len(list(domain.valid_actions(domain.solved_state())))

    # tree = SearchTree(domain, max_depth=2)
    # print(tree.layers)

    # tree = SearchTree(domain, max_depth=4) # 5 uses up 4+ GB memory
    # for depth in range(len(tree._layers)):
    #     print(len(tree._layers[depth]), A**depth)
    
    # tree = SearchTree(domain, max_depth=2)
    # for actions, permutation in tree:
    #     print(actions, permutation)

    # # iterate over search tree object and visually inspect states and their order, esp those more than 1 action away
    # tree = SearchTree(domain, max_depth=2)
    # import matplotlib.pyplot as pt
    # for n, (actions, permutation) in enumerate(tree):
    #     if n == 70: break
    #     state = domain.solved_state()[permutation]
    #     ax = pt.subplot(7, 10, n+1)
    #     domain.render(state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')
    #     ax.set_title(str(actions))
    # pt.show()

    # #### check whether color-neutral trees are isomorphic for any starting state
    # # the answer is NO.  also, not much smaller than non-neutral at shallow trees
    # # results (number of depth 4 tree nodes):
    # # solved_state, not neutral: 174604
    # # solved_state, yes neutral: 174491
    # # random_state(20), not neutral: 174604
    # # random_state(20), yes neutral: 174604 almost always, once was 174601
    # # random_state(1 or 2), yes neutral: around 174390
    # tree = SearchTree(domain, max_depth=4)
    # init = domain.random_state(2, np.random.default_rng())
    # # init = domain.solved_state()
    # color_neutral = True
    # # color_neutral = False

    # for rep in range(100):
    #     explored = set()
    #     cn = []
    #     for n, (actions, state) in enumerate(tree.rooted_at(init)):
    #         if color_neutral:
    #             recolorings = domain.recolorings_of(state)
    #             if any([state.tobytes() in explored for state in recolorings]): continue
    #         explored.add(state.tobytes())
    #         cn.append((actions, state))
    #     print(len(cn))

    # #### count distinct action sequences in tree
    # tree = SearchTree(domain, max_depth=3)
    # action_sequences = [a[1:] for a, _ in tree if len(a) > 1]
    # print(len(action_sequences))
    # print(len(set(action_sequences)))

    # #### count distinct action subsequences in tree
    # # this is also the number of distinct states in the tree, which makes sense in hindsight
    # import itertools as it
    # tree = SearchTree(domain, max_depth=4)
    # distinct = set()
    # repeated = 0
    # for actions, _ in tree:
    #     for lo,hi in it.combinations(range(len(actions)+1), 2):
    #         distinct.add(actions[lo:hi])
    #         repeated += 1
    # print(len(distinct))
    # print(repeated)

    # #### check frequencies and wildcards of each action subsequences in tree
    # # only lo (no hi) since hi < len(actions) is in a different branch to state reached at hi
    # # depth 4 results:
    # # distinct len-1 macros range from 5711 to 7125 occurences each,
    # # len-2 from 153 to 390, len 3 from 3 to 21, len 4 from 1 to 1 each
    # # number of invariant faces after distinct macros:
    # # almost constant for a fixed macro length, with a few outliers at larger depth
    # # len-1, len-2 have 0 invariant faces
    # # some len-3 macros have 27/54 invariant facies in their set of terminal states
    # # len-4 macros have 54 invariants but only because their sets of terminal states are singletons
    # import itertools as it
    # max_depth = 4
    # init = domain.random_state(20, np.random.default_rng())
    # # init = domain.solved_state()
    # tree = SearchTree(domain, max_depth)
    # distinct = {}
    # for actions, state in tree.rooted_at(init):
    #     for lo in range(len(actions)):
    #         if actions[lo:] not in distinct: distinct[actions[lo:]] = []
    #         distinct[actions[lo:]].append(state)

    # import matplotlib.pyplot as pt
    # data = [
    #     np.array([len(states) for macro, states in distinct.items() if len(macro) == k])
    #     for k in range(max_depth+1)]
    # for k in range(1,max_depth+1):
    #     print(k, data[k].min(), data[k].max(), data[k].mean())
    # pt.subplot(1, 2, 1)
    # pt.hist(data)
    # pt.xlabel("Occurrences of distinct macro")
    # pt.ylabel("Frequency")
    # pt.legend([str(k) for k in range(max_depth+1)])

    # data = [list() for _ in range(max_depth+1)]
    # for macro, states in distinct.items():
    #     state_array = np.array(states)
    #     invariants = (state_array == state_array[0]).all(axis=0)
    #     data[len(macro)].append(invariants.sum())
    # print("\nstate size = %d" % domain.state_size())
    # for k in range(1,max_depth+1):
    #     data[k] = np.array(data[k])
    #     print(k, data[k].min(), data[k].max(), data[k].mean())

    # pt.subplot(1, 2, 2)
    # pt.hist(data)
    # pt.xlabel("Number of invariants after distinct macro")
    # pt.ylabel("Frequency")
    # pt.legend([str(k) for k in range(max_depth+1)])

    # pt.show()

    #### profile
    import itertools as it
    valid_actions = tuple(it.product((0,1,2), (0,), (0, 1, 2, 3))) # only spinning one plane on each axis for 2cube

    from cube import CubeDomain
    domain = CubeDomain(2, valid_actions)
    init = domain.solved_state()

    tree = SearchTree(domain, 5)
    paths, states = zip(*tree.rooted_at(init))

    def prof(s):
        for _, neighbor in tree.rooted_at(states[s], up_to_depth=1):
            dumb = (np.arange(24) == np.arange(24*500).reshape(500, 24)).all(axis=1)

    for s in range(1000):
        # print(s)
        prof(s)

