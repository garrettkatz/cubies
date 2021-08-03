import pickle as pk
from pattern_database import PatternDatabase
from objectives import evaluate

def sample_macro(domain, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length):
    macro_size = rng.integers(min_macro_size, max_macro_size, endpoint=True)
    lo = rng.choice(rollout_length - macro_size)
    hi = lo + macro_size
    actions = rng.choice(list(domain.valid_actions()), size=rollout_length, replace=True)
    macro = domain.reverse(actions[lo:hi])
    state = domain.execute(actions[:hi], domain.solved_state())
    pattern = state * (rng.random(state.shape) > wildcard_rate).astype(int)
    return pattern, macro

def spawn(domain, num_patterns, sample_macro):
    patterns = []
    macros = []
    for p in range(num_patterns):
        pattern, macro = sample_macro()
        patterns.append(pattern)
        macros.append(macro)
    return PatternDatabase(patterns, macros, domain)

def mutate_random(domain, pattern_database, rng, mutation_rate, sample_macro):
    patterns = list(pattern_database.orig_patterns)
    macros = list(pattern_database.orig_macros)
    for p in range(len(patterns)):
        if rng.random() < mutation_rate:
            patterns[p], macros[p] = sample_macro()
    return PatternDatabase(patterns, macros, domain)

def mutate_unmatched(domain, pattern_database, sample_macro):
    patterns = list(pattern_database.orig_patterns)
    macros = list(pattern_database.orig_macros)
    match_counts = pattern_database.match_counts
    for p in np.flatnonzero(match_counts == match_counts.min()):
        patterns[p], macros[p] = sample_macro()
    return PatternDatabase(patterns, macros, domain)

def pareto_search(num_candidates, rng, spawn, mutate, evaluate, obj_names, dump_file):

    candidate = {}
    objective = np.empty((num_candidates, len(obj_names)))

    candidate[0], objective[0] = evaluate(spawn())
    frontier = np.array([0])

    for c in range(1, num_candidates):

        candidate[c], objective[c] = evaluate(mutate(candidate[rng.choice(frontier)]))

        dominators = (objective[frontier] > objective[c]).all(axis=1)
        remainders = (objective[frontier] >= objective[c]).any(axis=1)

        if not dominators.any(): frontier = np.append(frontier[remainders], [c])

        bests = ["%s: %s" % (obj_names[i], objective[:c+1, i].max()) for i in range(objective.shape[1])]
        print("%d  |  %d pioneers  |  bests: %s" % (c, frontier.size, ", ".join(bests)))
        
        if not dominators.any():
            with open(dump_file, "wb") as df: pk.dump((candidate, objective, frontier), df)
    
    return candidate, objective, frontier

if __name__ == "__main__":
    
    dotrain = True
    showresults = False
    postmortem = False

    # dotrain = False
    # showresults = True
    # postmortem = False

    # dotrain = False
    # showresults = False
    # postmortem = True

    # dotrain = True
    # showresults = True
    # postmortem = False

    cube_size = 3
    max_scrambles = 20
    num_instances = 64
    tree_depth = 3
    max_depth = 1
    max_macros = 5
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 5
    wildcard_rate = .5
    rollout_length = 20
    mutation_rate = .25
    num_candidates = 2**16
    # num_candidates = 256
    obj_names = ["macro size", "godly solves"]
    dump_file = "data.pkl"

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    solved = domain.solved_state()
    valid_actions = list(domain.valid_actions(solved))

    from tree import SearchTree
    bfs_tree = SearchTree(domain, tree_depth)

    import numpy as np
    rng = np.random.default_rng()

    sample_macro_fun = lambda: sample_macro(domain, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length)
    
    def evaluate_fun(candidate):
        candidate, objectives = evaluate(domain, bfs_tree, candidate, rng, num_instances, max_scrambles, max_depth, max_macros)
        objectives = np.array(objectives)[1:]
        return candidate, objectives

    # # test spawn
    # patterns, macros = spawn(num_patterns, sample_macro_fun)

    # # test mutate
    # patterns, macros = mutate(patterns, macros, rng, mutation_rate, sample_macro_fun)

    # pattern_database = PatternDatabase(patterns, macros, domain)
    # objectives = evaluate(domain, bfs_tree, pattern_database, rng, num_instances, max_scrambles, max_depth, max_macros)
    # pattern_size, macro_size, godly_solves = objectives
    # print(objectives)
    
    if dotrain:

        pareto_search(
            num_candidates,
            rng,
            spawn = lambda: spawn(domain, num_patterns, sample_macro_fun),
            # mutate = lambda candidate: mutate_random(domain, candidate, rng, mutation_rate, sample_macro_fun),
            mutate = lambda candidate: mutate_unmatched(domain, candidate, sample_macro_fun),
            evaluate = evaluate_fun,
            obj_names = obj_names,
            dump_file = dump_file,
        )

    import matplotlib.pyplot as pt

    if showresults:

        with open(dump_file, "rb") as f: (candidate, objectives, frontier) = pk.load(f)

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
        rando = objectives + .0*(rng.random(objectives.shape) - .5)
        
        pt.subplot(1,2,1)
        pt.scatter(*rando.T, color=color)
        # pt.scatter(*rando[frontier].T, color=color[frontier])

        pt.xlabel("- macro size")
        pt.ylabel("# godly solves")
        
        pt.subplot(1,2,2)
        pt.plot(np.arange(C), [candidate[c].match_counts.sum() for c in range(C)], '-k')
        pt.plot(frontier, [candidate[c].match_counts.sum() for c in frontier], '-ob')
        pt.xlabel("candidate")
        pt.ylabel("total match count")
        pt.show()

    if postmortem:
        with open(dump_file, "rb") as f: (candidate, objectives, frontier) = pk.load(f)
        
        match_counts = np.zeros(num_patterns)
        for f in range(len(frontier)):

            patterns, macros = candidate[frontier[f]]
            pdb = PatternDatabase(patterns, macros, domain)
    
            evaluate(domain, bfs_tree, pdb, rng, num_instances, max_scrambles, max_depth, max_macros),
            # # print(pdb.match_counts)
            match_counts += pdb.match_counts
            # print(pdb.match_counts.max())
            # print(pdb.hit_counts)
            print((pdb.patterns == 0).sum(axis=1), pdb.patterns.shape[1])
            
            # pt.subplot(4,1,f+1)
            # pt.imshow(pdb.patterns.T)
            # # pt.imshow(np.concatenate((pdb.hit_counts.T / pdb.num_queries, pdb.match_counts[np.newaxis]), axis=0))
            # if f == 3: break

            # mutate(patterns, macros, rng, mutation_rate, sample_macro_fun),

        pt.bar(np.arange(len(match_counts)), match_counts)

        pt.show()

