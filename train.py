from objectives import evaluate

if __name__ == "__main__":
    
    cube_size = 3
    max_scrambles = 5
    num_instances = 32
    tree_depth = 3
    max_depth = 1
    max_macros = 2
    num_patterns = 32
    unnormalized_wildcard_weight = 2
    population_size = 64
    num_generations = 64
    dotrain = True

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    solved = domain.solved_state()
    valid_actions = list(domain.valid_actions(solved))

    from tree import SearchTree
    bfs_tree = SearchTree(domain, tree_depth)

    import numpy as np
    rng = np.random.default_rng()
    probs = np.ones(7)
    probs[0] = unnormalized_wildcard_weight
    probs /= probs.sum()

    from pattern_database import PatternDatabase
    best_patterns = rng.choice(np.arange(7), size = (num_patterns, solved.size), p = probs)
    best_macros = [
        rng.choice(valid_actions, size = rng.integers(1,4), replace=True)
        for _ in range(num_patterns)]
    best_solves = 0

    import pickle as pk
    
    if dotrain:

        all_patterns = {}
        all_macros = {}
        all_godly_solves = {}
        for gen in range(num_generations):
        
            all_patterns[gen] = {}
            all_macros[gen] = {}
            all_godly_solves[gen] = {}
            for rep in range(population_size):
    
                patterns = best_patterns.copy()
                macros = [list(macro) for macro in best_macros]
                for m in range(len(macros)):
                    mutators = np.flatnonzero(rng.random(domain.state_size()) < 0.05)
                    patterns[m, mutators] = rng.choice(np.arange(7), size = mutators.size, p = probs)
                    mutators = np.flatnonzero(rng.random(len(macros[m])) < 0.1)
                    for a in mutators: macros[m][a] = rng.choice(valid_actions)
                
                pattern_database = PatternDatabase(patterns, macros, domain)
    
                objectives = evaluate(domain, bfs_tree, pattern_database, rng, num_instances, max_scrambles, max_depth, max_macros)
                pattern_size, macro_size, godly_solves = objectives
                
                all_patterns[gen][rep] = patterns
                all_macros[gen][rep] = macros
                all_godly_solves[gen][rep] = godly_solves
                with open("data.pkl", "wb") as f: pk.dump((all_patterns, all_macros, all_godly_solves), f)
    
                print("  rep %d: %d godly solves" % (rep, godly_solves))
            
            best = np.argmax([all_godly_solves[gen][rep] for rep in range(population_size)])
            if all_godly_solves[gen][best] > best_solves:
                best_solves = all_godly_solves[gen][best]
                best_patterns = all_patterns[gen][best]
                best_macros = all_macros[gen][best]
            print("gen %d: best godly solves = %d" % (gen, best_solves))

    with open("data.pkl", "rb") as f: (patterns, macros, godly_solves) = pk.load(f)
    
    import matplotlib.pyplot as pt
    pt.subplot(1,2,1)
    pt.hist(godly_solves[0].values())
    pt.hist(godly_solves[len(godly_solves)-1].values())
    pt.subplot(1,2,2)
    pt.plot([max(godly_solves[gen].values()) for gen in range(len(godly_solves))])
    pt.show()


