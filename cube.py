"""
NxNxN rubiks cube
state is a NxNxNx3 array
first 3 dimensions are positions on the cube
last dimension is the colors in each spatial direction
spatial directions are 0:x, 1:y, 2:z
"""
import numpy as np
import matplotlib.pyplot as pt
import itertools as it
from matplotlib.patches import Polygon

# Set up color enum and rgb tuples
_R, _G, _B, _W, _Y, _O = range(1,7)
_colors = {
    _R: (1.0, 0.0, 0.0), # red
    _G: (0.0, 1.0, 0.0), # green
    _B: (0.0, 0.0, 1.0), # blue
    _W: (1.0, 1.0, 1.0), # white
    _Y: (1.0, 1.0, 0.0), # yellow
    _O: (1.0, 0.6, 0.0), # orange
}

class CubeDomain:

    def __init__(self, N):
        # N is side-length of cube
        
        # Count cubies and facies
        num_facies = 6*N**2
        num_cubies = N**3 * 3

        # Build solved_cube
        # solved_cube[i,j,k,d] is color of facie at cubie position (i,j,k) normal to d^th rotation axis
        solved_cube = np.zeros((N,N,N,3), dtype=int)
        solved_cube[ 0, :, :,0] = _R
        solved_cube[-1, :, :,0] = _O
        solved_cube[ :, 0, :,1] = _W
        solved_cube[ :,-1, :,1] = _Y
        solved_cube[ :, :, 0,2] = _B
        solved_cube[ :, :,-1,2] = _G

        # state representation is flat array of external facie colors only
        # cube_index[i,j,k,d] is index in state array of facie at position (i,j,k) normal to d^th axis
        # for internal facies, cube_index[i,j,k,d] == -1
        # face_index is the partial inverse of cube_index.  For the i^th facie in the state representation:
        # cube_index.flat[face_index[i]] == i
        
        # Make external facies non-zero in cube_index
        cube_index = np.zeros((N,N,N,3), dtype=int)
        cube_index[ 0, :, :,0] = 1
        cube_index[-1, :, :,0] = 1
        cube_index[ :, 0, :,1] = 1
        cube_index[ :,-1, :,1] = 1
        cube_index[ :, :, 0,2] = 1
        cube_index[ :, :,-1,2] = 1
        
        # Finalize index data from non-zero facies
        face_index = np.flatnonzero(cube_index)
        cube_index[:] = -1 # for all internal elements
        cube_index.flat[face_index] = np.arange(num_facies) # overwrite external facie indices

        # flatten the solved cube to state representation using face_index
        solved_state = solved_cube.flat[face_index]
        
        # twists are performed by permuting indices in the state array
        # new_state = old_state[twist_permutation[a, p, n]]
        # twist_permutation[a, p, n] is the permuted index array for n quarter twists of plane p around axis a
        twist_permutation = np.empty((3, N, 4, num_facies), dtype=int)

        # compute quarter twists for each axis
        for p in range(N):
            # rotx
            permuted = cube_index.copy()
            permuted[p,:,:,:] = np.rot90(permuted[p,:,:,:], axes=(0,1)) # rotate cubie positions
            permuted[p,:,:,(1,2)] = permuted[p,:,:,(2,1)] # rotate cubies
            twist_permutation[0, p, 1] = permuted.flat[face_index]
            # roty
            permuted = cube_index.copy()
            permuted[:,p,:,:] = np.rot90(permuted[:,p,:,:], axes=(0,1))
            permuted[:,p,:,(2,0)] = permuted[:,p,:,(0,2)]
            twist_permutation[1, p, 1] = permuted.flat[face_index]
            # rotz
            permuted = cube_index.copy()
            permuted[:,:,p,:] = np.rot90(permuted[:,:,p,:], axes=(0,1))
            permuted[:,:,p,(0,1)] = permuted[:,:,p,(1,0)]
            twist_permutation[2, p, 1] = permuted.flat[face_index]
        # compute non-quarter twists
        for a, p, n in it.product((0,1,2), range(N), (2, 3, 4)):
            twist_permutation[a, p, n % 4] = twist_permutation[a, p, n-1][twist_permutation[a, p, 1]]

        # rotational symmetries of the full cube are also computed via state permutations
        symmetry_permutation = np.empty((24, num_facies), dtype=int)

        # helper function to rotate all planes around a given axis
        def rotate_all_planes(state, axis, num_twists):
            state = state.copy()
            for plane in range(N):
                state = state[twist_permutation[axis, plane, num_twists % 4]]
            return state

        # compute symmetry permutations
        for s, (axis, direction, num_twists) in enumerate(it.product((0,1,2),(-1,1),(0,1,2,3))):
            # align top face with one of six directed axes
            permuted = np.arange(num_facies)
            if axis != 2: permuted = rotate_all_planes(permuted, 1-axis, direction)
            elif direction != 1: permuted = rotate_all_planes(permuted, 0, 2)
            # rotate cube around directed axis
            permuted = rotate_all_planes(permuted, axis, num_twists)
            symmetry_permutation[s] = permuted

        # determine symmetry inverses
        inverse_symmetry = {}
        for s, s_inv in it.product(range(24), repeat=2):
            if (symmetry_permutation[s_inv][symmetry_permutation[s]] == np.arange(num_facies)).all():
                inverse_symmetry[s] = s_inv

        # physically possible permutations of the colors correspond to full cube symmetries
        color_permutation = np.zeros((24, 7), dtype=int) # 7 since color enum starts at 1
        
        # get one index in solved state for each color
        color_index = np.array([(solved_state == c).argmax() for c in range(1,7)])
        
        # extract color permutation from each symmetry permutation
        for sym in range(24):
            color_permutation[sym,1:] = solved_state[symmetry_permutation[sym]][color_index]

        # precompute valid action list
        # action format: (rotation_axis, plane_index, num_twists)
        valid_actions = tuple(it.product((0,1,2), range(N), (1,2,3)))

        # precompute symmetries of solved state
        solved_states = solved_state[symmetry_permutation].copy()

        # memoize results
        self.N = N
        self._face_index = face_index
        self._solved_state = solved_state
        self._solved_states = solved_states
        self._twist_permutation = twist_permutation
        self._symmetry_permutation = symmetry_permutation
        self._inverse_symmetry = inverse_symmetry
        self._color_permutation = color_permutation
        self._valid_actions = valid_actions
    
    def god_number(self):
        return 11 if self.N == 2 else 20

    def state_size(self):
        return self._solved_state.size

    def solved_state(self):
        return self._solved_state.copy()

    def valid_actions(self, state=None):
        # action format: (rotation_axis, plane_index, num_twists)
        return self._valid_actions

    def perform(self, action, state):
        axis, plane, num_twists = action
        return state[self._twist_permutation[axis, plane, num_twists % 4]].copy()

    def execute(self, actions, state):
        for action in actions: state = self.perform(action, state)
        return state

    def is_solved_in(self, state):
        return (self._solved_states == state).all(axis=1).any()

    def symmetries_of(self, state):
        return state[self._symmetry_permutation].copy()

    def inverse_symmetry_of(self, s):
        return self._inverse_symmetry[s]

    def color_permutations_of(self, state):
        return self._color_permutation.take(state, axis=1)

    def reverse(self, actions):
        return [(axis, plane, -twists % 4) for (axis, plane, twists) in reversed(actions)]

    def superflip_path(self):
        # from https://www.cube20.org
        path = "R L U2 F U' D F2 R2 B2 L U2 F' B' U R2 D F2 U R2 U"
        action_map = {
            "U": (1, 0, 3),
            "D": (1, self.N-1, 1),
            "L": (2, self.N-1, 3),
            "R": (2, 0, 1),
            "F": (0, 0, 1),
            "B": (0, self.N-1, 3),
            "U2": (1, 0, 2),
            "D2": (1, self.N-1, 2),
            "L2": (2, self.N-1, 2),
            "R2": (2, 0, 2),
            "F2": (0, 0, 2),
            "B2": (0, self.N-1, 2),
            "U'": (1, 0, 1),
            "D'": (1, self.N-1, 3),
            "L'": (2, self.N-1, 1),
            "R'": (2, 0, 3),
            "F'": (0, 0, 3),
            "B'": (0, self.N-1, 1),
        }
        return [action_map[a] for a in path.split(" ")]

    def random_state(self, scramble_length, rng):
        state = self.solved_state()
        valid_actions = tuple(self.valid_actions(state))
        for s in range(scramble_length):
            state = self.perform(rng.choice(valid_actions), state)
        return state

    def render(self, state, ax, x0=0, y0=0):
        # ax is matplotlib Axes object
        # unflatten state into cube for easier indexing
        N = self.N
        cube = np.empty((N,N,N,3), dtype=int)
        cube.flat[self._face_index] = state
        # render orthogonal projection
        angles = -np.arange(3) * np.pi * 2 / 3
        axes = np.array([np.cos(angles), np.sin(angles)])
        for d in range(3):
            for a, b in it.product(range(N), repeat=2):
                xy = [ a   *axes[:,d] +  b   *axes[:,(d+1) % 3],
                      (a+1)*axes[:,d] +  b   *axes[:,(d+1) % 3],
                      (a+1)*axes[:,d] + (b+1)*axes[:,(d+1) % 3],
                       a   *axes[:,d] + (b+1)*axes[:,(d+1) % 3]]
                xy = [(x+x0, y+y0) for (x,y) in xy]
                c = _colors[cube[tuple(np.roll((a,b,0),d))+((d+2) % 3,)]]
                ax.add_patch(Polygon(xy, facecolor=c, edgecolor='k'))
            ax.text((N+.1)*axes[0,d], (N+.1)*axes[1,d], str(d))

if __name__ == "__main__":


    # #### test performing actions
    # domain = CubeDomain(4)
    # actions = [(1, 0, 1), (0, 1, 1), (2, 2, 1), (1, 0, 1)]
    # # actions = [(0,0,1)]
    # state = domain.solved_state()

    # ax = pt.subplot(1, len(actions)+1, 1)
    # domain.render(state, ax, 0, 0)
    # ax.axis("equal")
    # ax.axis('off')
    
    # for a, (axis, depth, num) in enumerate(actions):
    #     state = domain.perform((axis, depth, num), state)

    #     ax = pt.subplot(1, len(actions)+1, a+2)
    #     domain.render(state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')
    
    # pt.show()

    # #### test symmetries
    # domain = CubeDomain(3)
    # state = domain.solved_state()
    # # state = domain.perform((0, 0, 1), state)
    # for s, sym_state in enumerate(domain.symmetries_of(state)):
    #     ax = pt.subplot(4, 6, s+1)
    #     domain.render(sym_state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')
    #     ax.set_title(str(s))
    # pt.show()

    # #### test color permutations
    # domain = CubeDomain(2)
    # print(domain._color_permutation)
    # state = domain.solved_state()
    # # state = domain.perform((0, 0, 1), state)
    # for s, sym_state in enumerate(domain.color_permutations_of(state)):
    # # for s in range(24):
    # #     sym_state = domain._color_permutation[s][state]
    #     ax = pt.subplot(4, 6, s+1)
    #     domain.render(sym_state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')
    #     ax.set_title(str(s))
    # pt.show()

    #### test hardest state
    domain = CubeDomain(3)
    path = domain.superflip_path() # from unsolved to solved
    # inverted = [a[:2]+(-a[2] % 4,) for a in path[::-1]] # from solved to unsolved
    hardest_state = domain.execute(domain.reverse(path), domain.solved_state())
    states = [hardest_state]
    for action in path: states.append(domain.perform(action, states[-1]))
    assert domain.is_solved_in(states[-1])
    for s, state in enumerate(states):
        ax = pt.subplot(4, 6, s+1)
        domain.render(state, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
    pt.show()
