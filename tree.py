"""
Memoize a depth-limited, BFS search tree for the cube domain
Paths and resulting permutations to each node are precomputed
"""
import numpy as np

class SearchTree:

    def __init__(self, domain, max_depth, orientation_neutral=True, color_neutral=True):

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
                    if new_permutation.tobytes() in explored: continue
                    
                    # or if a reorientation already explored and neutral
                    if orientation_neutral:
                        orientations = domain.orientations_of(new_permutation)
                        if any([perm.tobytes() in explored for perm in orientations]): continue

                    # or if a recoloring already explored and neutral
                    if color_neutral:
                        # TODO: fix here and cube.py for permuting facie indices instead of color enum
                        recolorings = domain.recolorings_of(new_permutation)
                        if any([perm.tobytes() in explored for perm in recolorings]): continue

                    # otherwise add the new state to the frontier and explored set
                    explored.add(new_permutation.tobytes())
                    layers[depth+1].append((new_actions, new_permutation))
        
        self._layers = layers
    
    def __iter__(self):
        for depth in range(len(self._layers)):
            for actions, permutation in self._layers[depth]:
                yield actions, permutation

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

    # iterate over search tree object and visually inspect states and their order, esp those more than 1 action away
    tree = SearchTree(domain, max_depth=2)
    import matplotlib.pyplot as pt
    for n, (actions, permutation) in enumerate(tree):
        if n == 70: break
        state = domain.solved_state()[permutation]
        ax = pt.subplot(7, 10, n+1)
        domain.render(state, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
        ax.set_title(str(actions))
    pt.show()

    
 

