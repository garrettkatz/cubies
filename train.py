
if __name__ == "__main__":
    
    from cube import CubeDomain
    domain = CubeDomain(3)
    scramble_length = 100
    
    import random
    states = [domain.solved_state()]
    actions = []
    for s in range(scramble_length):
        action = random.choice(list(domain.valid_actions(states[s])))
        state = domain.perform(action, states[s])
        actions.append(action)
        states.append(state)
    
    import matplotlib.pyplot as pt
    for s, state in enumerate(states):
        ax = pt.subplot(10, 11, s+1)
        domain.render(state, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
    pt.show()

