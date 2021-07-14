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

def solved_state(N):
    solved = np.zeros((N,N,N,3))
    solved[ 0, :, :,0] = _R
    solved[-1, :, :,0] = _O
    solved[ :, 0, :,1] = _W
    solved[ :,-1, :,1] = _Y
    solved[ :, :, 0,2] = _B
    solved[ :, :,-1,2] = _G
    return solved
def is_solved_in(state):
    return \
        (state[ 0, :, :,0] == _R).all() and \
        (state[-1, :, :,0] == _O).all() and \
        (state[ :, 0, :,1] == _W).all() and \
        (state[ :,-1, :,1] == _Y).all() and \
        (state[ :, :, 0,2] == _B).all() and \
        (state[ :, :,-1,2] == _G).all()

def rotx_(state, depth, num_turns):
    state[depth,:,:,:] = np.rot90(state[depth,:,:,:], k=num_turns, axes=(0,1))
    if num_turns % 2 == 1: state[depth,:,:,(1,2)] = state[depth,:,:,(2,1)]
def roty_(state, depth, num_turns):
    state[:,depth,:,:] = np.rot90(state[:,depth,:,:], k=num_turns, axes=(0,1))
    if num_turns % 2 == 1: state[:,depth,:,(2,0)] = state[:,depth,:,(0,2)]
def rotz_(state, depth, num_turns):
    state[:,:,depth,:] = np.rot90(state[:,:,depth,:], k=num_turns, axes=(0,1))
    if num_turns % 2 == 1: state[:,:,depth,(0,1)] = state[:,:,depth,(1,0)]

def perform_(action, state):
    rot_, depth, num_turns = action
    rot_(state, depth, num_turns)
    return state
def perform(action, state):    
    return perform_(action, state.copy())
def valid_actions(state):
    N = state.shape[0]
    return it.product((rotx_, roty_, rotz_), range(N), (-3,-2,-1,1,2,3))

def render(ax, state, x0=0, y0=0):
    angles = -np.arange(3) * np.pi * 2 / 3
    axes = np.array([np.cos(angles), np.sin(angles)])
    N = state.shape[0]
    for d in range(3):
        for a, b in it.product(range(N), repeat=2):
            xy = [ a   *axes[:,d] +  b   *axes[:,(d+1) % 3],
                  (a+1)*axes[:,d] +  b   *axes[:,(d+1) % 3],
                  (a+1)*axes[:,d] + (b+1)*axes[:,(d+1) % 3],
                   a   *axes[:,d] + (b+1)*axes[:,(d+1) % 3]]
            xy = [(x+x0, y+y0) for (x,y) in xy]
            c = _colors[state[tuple(np.roll((a,b,0),d))+((d+2) % 3,)]]
            ax.add_patch(Polygon(xy, facecolor=c, edgecolor='k'))

class PatternDatabase:
    def __init__(self):
        pass
    def query(self, state):
        return False
    def result(self):
        return []

if __name__ == "__main__":

    ax = pt.gca()
    state = solved_state(3)
    # state = do_action(state, 1, 0, 3)
    rotz_(state, 0, 1)
    rotx_(state, 0, 1)

    render(ax, state, 0, 0)
    ax.axis("equal")
    ax.axis('off')
    
    pt.show()
