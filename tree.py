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

    def rooted_at(self, state):
        for depth in range(len(self._layers)):
            for actions, permutation in self._layers[depth]:
                yield actions, state[permutation]


if __name__ == "__main__":
    
    from cube import CubeDomain
    domain = CubeDomain(3)
    A = len(list(domain.valid_actions(domain.solved_state())))

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

    #### check whether color-neutral trees are isomorphic for any starting state
    # the answer is NO
    # results (number of depth 4 tree nodes):
    # solved_state, not neutral: 174604
    # solved_state, yes neutral: 174491
    # random_state(20), not neutral: 174604
    # random_state(20), yes neutral: 174604 almost always, once was 174601
    # random_state(1 or 2), yes neutral: around 174390
    tree = SearchTree(domain, max_depth=4)
    init = domain.random_state(2, np.random.default_rng())
    # init = domain.solved_state()
    color_neutral = True
    # color_neutral = False

    for rep in range(100):
        explored = set()
        cn = []
        for n, (actions, state) in enumerate(tree.rooted_at(init)):
            if color_neutral:
                recolorings = domain.recolorings_of(state)
                if any([state.tobytes() in explored for state in recolorings]): continue
            explored.add(state.tobytes())
            cn.append((actions, state))
        print(len(cn))

