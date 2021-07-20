"""
Memoize a depth-limited, BFS search tree for the cube domain
Paths and resulting permutations to each node are precomputed
"""
import numpy as np

class SearchTree:
    def __init__(self, domain, max_depth):
        permutation = np.arange(len(domain.solved_state()))
        actions = tuple()
        
        explored = set([permutation.tobytes()])
        layers = {0: [(actions, permutation)]}
        
        for depth in range(max_depth):
            layers[depth+1] = []

            for actions, permutation in layers[depth]:
                for action in domain.valid_actions(permutation):

                    new_actions = actions + (action,)
                    new_permutation = domain.perform(action, permutation)

                    if new_permutation.tobytes() not in explored:
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

    
 

