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
        
        # quarter twists are performed by permuting indices in the state array
        # new_state = old_state[twist_permutation[a, p]]
        # twist_permutation[a, p] is the permuted index array for one quarter twist of plane p along axis a

        twist_permutation = np.empty((3, N, num_facies), dtype=int)
        for p in range(N):
            # rotx
            permuted = cube_index.copy()
            permuted[p,:,:,:] = np.rot90(permuted[p,:,:,:], axes=(0,1)) # rotate cubie positions
            permuted[p,:,:,(1,2)] = permuted[p,:,:,(2,1)] # rotate cubies
            twist_permutation[0, p] = permuted.flat[face_index]
            # roty
            permuted = cube_index.copy()
            permuted[:,p,:,:] = np.rot90(permuted[:,p,:,:], axes=(0,1))
            permuted[:,p,:,(2,0)] = permuted[:,p,:,(0,2)]
            twist_permutation[1, p] = permuted.flat[face_index]
            # rotz
            permuted = cube_index.copy()
            permuted[:,:,p,:] = np.rot90(permuted[:,:,p,:], axes=(0,1))
            permuted[:,:,p,(0,1)] = permuted[:,:,p,(1,0)]
            twist_permutation[2, p] = permuted.flat[face_index]
        
        # # symmetries are also computed via face permutations
        # def rotate(state, axis, num_twists):
        #     for plane in range(N): state = self.perform((axis, plane, num_twists), state)
        #     return state
        # symmetry_permutation = np.empty((24, num_facies), dtype=int)
        # for axis, direction in it.product((0,1,2),(0,1)):
        #     for num_twists in range(4):
        #         permuted = cube_index.copy()
        #         if axis > 0: permuted = rotate(permuted, axis, 1)
        #         if direction > 0: permuted = rotate(permuted, axis, 2)
        #         permuted = rotate(permuted, axis, k = num_twists * direction)
        
        # memoize results
        self.N = N
        self.face_index = face_index
        self._solved_state = solved_state
        self.twist_permutation = twist_permutation
    
    def solved_state(self):
        return self._solved_state.copy()
    
    def valid_actions(self, state):
        # action format: (rotation_axis, plane_index, num_twists)
        N = self.N
        return it.product((0,1,2), range(N), (1,2,3))

    def perform(self, action, state):
        rotation_axis, plane_index, num_twists = action
        state = state.copy()
        for k in range(num_twists):
            state = state[self.twist_permutation[rotation_axis, plane_index]]
        return state

    def is_solved_in(self, state):
        return (state == self._solved_state).all()

    # def states_symmetric_to(self, state):
    #     symmetries = (
    #         (),
    #         ((0,0,1),(0,1,1),(0,
    #         0,1)),
    #     )
    #     for axis, direction in it.product((0,1,2), (-1,1)):

    def render(self, state, ax, x0=0, y0=0):
        # ax is matplotlib Axes object
        # unflatten state into cube for easier indexing
        N = self.N
        cube = np.empty((N,N,N,3), dtype=int)
        cube.flat[self.face_index] = state
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

if __name__ == "__main__":


    domain = CubeDomain(4)
    actions = [(1, 0, 1), (0, 1, 1), (2, 2, 1), (1, 0, 1)]
    # actions = [(0,0,1)]
    state = domain.solved_state()

    ax = pt.subplot(1, len(actions)+1, 1)
    domain.render(state, ax, 0, 0)
    ax.axis("equal")
    ax.axis('off')
    
    for a, (axis, depth, num) in enumerate(actions):
        state = domain.perform((axis, depth, num), state)

        ax = pt.subplot(1, len(actions)+1, a+2)
        domain.render(state, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
    
    pt.show()
