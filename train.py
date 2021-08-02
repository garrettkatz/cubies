import pickle as pk
from objectives import evaluate

def sample_macro(domain, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length):
    macro_size = rng.integers(min_macro_size, max_macro_size, endpoint=True)
    lo = rng.choice(rollout_length - macro_size)
    hi = lo + macro_size
    actions = rng.choice(list(domain.valid_actions()), size=rollout_length, replace=True)
    macro = domain.reverse(actions[lo:hi])
    state = domain.execute(actions[:hi], domain.solved_state())
    pattern = state * (rng.random(state.shape) < wildcard_rate).astype(int)
    return pattern, macro

def spawn(num_patterns, sample_macro):
    patterns = []
    macros = []
    for p in range(num_patterns):
        pattern, macro = sample_macro()
        patterns.append(pattern)
        macros.append(macro)
    return patterns, macros

def mutate(patterns, macros, rng, mutation_rate, sample_macro):
    patterns = list(patterns)
    macros = list(macros)
    for p in range(len(patterns)):
        if rng.random() < mutation_rate:
            patterns[p], macros[p] = sample_macro()
    return patterns, macros

def optimize(num_generations, num_candidates, spawn, mutate, evaluate, dump_file):
    # initialize populations
    candidate = {g: {} for g in range(num_generations)}
    objectives = {g: {} for g in range(num_generations)}

    # optimize
    for g in range(num_generations):
        for c in range(num_candidates):
            new_candidate = spawn() if g == 0 else mutate(candidate[g-1][c])
            new_objectives = evaluate(new_candidate)
            # if g == 0 or (new_objectives > objectives[g-1][c]).any():
            if g == 0 or (new_objectives[2] > objectives[g-1][c][2]).any():
                candidate[g][c] = new_candidate
                objectives[g][c] = new_objectives
            else:
                candidate[g][c] = candidate[g-1][c]
                objectives[g][c] = objectives[g-1][c]
            pattern_size, macro_size, godly_solves = objectives[g][c]
            print("  g,c = %d,%d: %d size, %d godly" % (g,c, macro_size, godly_solves))

        with open(dump_file, "wb") as df: pk.dump((candidate, objectives), df)

        gen_objectives = np.array(list(objectives[g].values()))
        best_size, best_godly = gen_objectives[:,1:].max(axis=0)
        print("Gen %d: best size, godly = %d, %d" % (g, best_size, best_godly))

    return candidate, objectives

if __name__ == "__main__":
    
    dotrain = True
    showresults = True

    cube_size = 3
    max_scrambles = 20
    num_instances = 64
    tree_depth = 3
    max_depth = 1
    max_macros = 2
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 5
    wildcard_rate = .5
    rollout_length = 20
    mutation_rate = .25
    num_generations = 512
    num_candidates = 64
    # num_generations = 3
    # num_candidates = 3
    dump_file = "data.pkl"

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    solved = domain.solved_state()
    valid_actions = list(domain.valid_actions(solved))

    from tree import SearchTree
    bfs_tree = SearchTree(domain, tree_depth)

    import numpy as np
    rng = np.random.default_rng()

    from pattern_database import PatternDatabase

    sample_macro_fun = lambda: sample_macro(domain, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length)

    # # test spawn
    # patterns, macros = spawn(num_patterns, sample_macro_fun)

    # # test mutate
    # patterns, macros = mutate(patterns, macros, rng, mutation_rate, sample_macro_fun)

    # pattern_database = PatternDatabase(patterns, macros, domain)
    # objectives = evaluate(domain, bfs_tree, pattern_database, rng, num_instances, max_scrambles, max_depth, max_macros)
    # pattern_size, macro_size, godly_solves = objectives
    # print(objectives)
    
    if dotrain:
        # candidate = (patterns, macros)
        result = optimize(
            num_generations,
            num_candidates,
            spawn = lambda: spawn(num_patterns, sample_macro_fun),
            mutate = lambda candidate: mutate(candidate[0], candidate[1], rng, mutation_rate, sample_macro_fun),
            evaluate = lambda candidate: np.array(evaluate(
                domain, bfs_tree, PatternDatabase(candidate[0], candidate[1], domain), rng, num_instances, max_scrambles, max_depth, max_macros)),
            dump_file = dump_file
        )

    if showresults:

        with open(dump_file, "rb") as f: (candidate, objectives) = pk.load(f)
        
        import matplotlib.pyplot as pt

        num_generations = len(objectives)
        num_candidates = len(objectives[0])
        num_finished = sum([len(objectives[g]) > 0 for g in range(num_generations)])
        objectives = np.array([[objectives[g][c] for c in range(num_candidates)] for g in range(num_finished)]).astype(float)
        
        for c in range(min(1,num_candidates)):
            macro_sizes, godly_solves = objectives[:,c,1:].T
            for g in range(num_finished-1):
                color = (1 - (g+1) / num_finished,)*3
                assert (macro_sizes[g+1] >= macro_sizes[g]) or (godly_solves[g+1] >= godly_solves[g])
                pt.plot(macro_sizes[g:g+2], godly_solves[g:g+2], color=color, linestyle="-")
            # color = [(1 - (g+1) / num_finished,)*3 for g in range(num_finished)]
            # pt.scatter(macro_sizes, godly_solves, color=color)

        # for g in range(num_finished):
        #     gen_objectives = objectives[g]
        #     gen_objectives += (rng.random(gen_objectives.shape) - 0.5)
        #     macro_sizes, godly_solves = gen_objectives[:,1:].T
        #     color = (1 - (g+1) / num_finished,)*3
        #     pt.scatter(macro_sizes, godly_solves, color=color)

        pt.xlabel("-macro size")
        pt.ylabel("\# godly solves")
        pt.show()

