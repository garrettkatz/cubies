"""
solve in <= 20 steps
solve in <= scramble_length steps
smaller number of patterns
smaller macro lengths
less complex patterns (like colors grouped together, more zeros, fewer variables)
"""

if __name__ == "__main__":
    
    cube_size = 3
    max_scramble_length = 5
    num_instances = 128
    max_depth = 3
    num_initial_patterns = 32

    from cube import CubeDomain
    domain = CubeDomain(cube_size)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    import numpy as np
    rng = np.random.default_rng()

    from pattern_database import PatternDatabase
    solved = domain.solved_state()
    valid_actions = list(domain.valid_actions(solved))
    probs = np.ones(7)
    probs[0] = 10
    probs /= probs.sum()
    patterns = rng.choice(np.arange(7), size = (num_initial_patterns, solved.size), p = probs)
    macros = [
        rng.choice(valid_actions, size = rng.integers(1,4), replace=True)
        for _ in range(num_initial_patterns)]
    pattern_database = PatternDatabase(patterns, macros, domain)
    
    from algorithm import run
    
    result = {}
    scramble_length = {}
    plan_length = {}
    for i in range(num_instances):
        scramble_length[i] = rng.integers(1, max_scramble_length, endpoint=True)
        state = domain.random_state(scramble_length[i], rng)
        result[i] = run(state, domain, bfs_tree, pattern_database, max_depth=1, max_macros=2)
        print(i, result[i])
        
        if result[i] != False:
            plan_length[i] = 0
            for (actions, sym, macro) in result[i]: plan_length[i] += len(actions) + len(macro)

    # smaller number of patterns
    num_patterns = pattern_database.patterns.shape[0] / 24

    # smaller macro lengths
    macro_size = sum(map(len, pattern_database.macros)) / 24

    # TODO: less complex patterns (like colors grouped together/symmetric, more zeros, fewer variables)

    # solve in <= 20 steps
    godly_solves = sum(int((result[i] != False) and (plan_length[i] <= 20)) for i in range(num_instances))
    
    # solve in <= scramble_length steps
    scramble_solves = sum(int((result[i] != False) and (plan_length[i] <= scramble_length[i])) for i in range(num_instances))

    print()
    print("num_patterns = %d" % num_patterns)
    print("macro_size = %d" % macro_size)
    print("godly_solves = %d" % godly_solves)
    print("scramble_solves = %d" % scramble_solves)

