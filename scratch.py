if __name__ == "__main__":

    # check how many interstates of scramble are solvable by algorithm
    from cube import CubeDomain
    from pattern_database import PatternDatabase
    domain = CubeDomain(2)
    state = domain.solved_state()
    patterns, wildcard, macros =  [],[],[]
    pattern_database = PatternDatabase(patterns, wildcard, macros, domain)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth=4)

    import numpy as np
    rng = np.random.default_rng()
    valid_actions = tuple(domain.valid_actions(state))
    states = []
    actions = []
    for s in range(20):
        action = rng.choice(valid_actions)
        state = domain.perform(action, state)
        states.append(state)
        actions.append(action)

    from algorithm import run
    num_action = []
    was_solved = []
    for state in states:
        solved, plan = run(state, domain, bfs_tree, pattern_database, max_depth=4, max_macros=5)
        was_solved.append(solved)
        num_action.append(sum([len(a)+len(m) for (a,_,m) in plan]))
    
    for (a, s, n) in zip(actions, was_solved, num_action):
        print(a, n, s)
        
