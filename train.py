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

def pareto_search(num_candidates, rng, spawn, mutate, evaluate, obj_names, dump_file):

    objective = np.empty((num_candidates, len(obj_names)))
    candidate = {0: spawn()}
    objective[0] = evaluate(candidate[0])
    frontier = np.array([0])

    for c in range(1, num_candidates):

        candidate[c] = mutate(candidate[rng.choice(frontier)])
        objective[c] = evaluate(candidate[c])

        # comparison = (objective[frontier] > objective[c])
        # dominators = comparison.all(axis=1)
        # remainders = comparison.any(axis=1)
        dominators = (objective[frontier] > objective[c]).all(axis=1)
        remainders = (objective[frontier] >= objective[c]).any(axis=1)

        if not dominators.any(): frontier = np.append(frontier[remainders], [c])

        bests = ["%s: %s" % (obj_names[i], objective[:c+1, i].max()) for i in range(objective.shape[1])]
        print("%d  |  %d pioneers  |  bests: %s" % (c, frontier.size, ", ".join(bests)))
        
        with open(dump_file, "wb") as df: pk.dump((candidate, objective, frontier), df)
    
    return candidate, objective, frontier

if __name__ == "__main__":
    
    dotrain = True
    showresults = False
    # dotrain = False
    # showresults = True

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
    num_candidates = 2**16
    # num_candidates = 2**12
    obj_names = ["num patterns", "macro size", "godly solves"]
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
        pareto_search(
            num_candidates,
            rng,
            spawn = lambda: spawn(num_patterns, sample_macro_fun),
            mutate = lambda candidate: mutate(candidate[0], candidate[1], rng, mutation_rate, sample_macro_fun),
            evaluate = lambda candidate: np.array(evaluate(
                domain, bfs_tree, PatternDatabase(candidate[0], candidate[1], domain), rng, num_instances, max_scrambles, max_depth, max_macros)[1:]),
            obj_names = obj_names[1:],
            dump_file = dump_file,
        )

    if showresults:

        with open(dump_file, "rb") as f: (candidate, objectives, frontier) = pk.load(f)
        
        import matplotlib.pyplot as pt

        C = len(candidate)
        objectives = objectives[:C]
        color = np.tile(np.linspace(1, .5, C), (3,1)).T
        color[frontier,:] = 0
        # # color = np.ones((C, 3))
        # # color[:,0] = np.linspace(0, .5, C)
        # # color[frontier,2] = color[frontier, 0]
        # # color[frontier,0] = 1
        # color = np.zeros((C, 3))
        # color[:,0] = 1
        # color[frontier,2] = color[frontier, 0]
        # color[frontier,0] = 0
        rando = objectives + 0.25*(rng.random(objectives.shape) - .5)
        pt.scatter(*rando.T, color=color)
        # pt.scatter(*rando[frontier].T, color=color[frontier])

        pt.xlabel("- macro size")
        pt.ylabel("# godly solves")
        pt.show()

