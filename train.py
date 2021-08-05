import pickle as pk

def pareto_search(num_candidates, rng, spawn, mutate, evaluate, obj_names, dump_file):

    candidate = {}
    objective = np.empty((num_candidates, len(obj_names)))

    candidate[0], objective[0] = evaluate(spawn())
    frontier = np.array([0])
    pioneer = dict(candidate) # candidates that were ever in a frontier

    for c in range(1, num_candidates):

        candidate[c], objective[c] = evaluate(mutate(candidate[rng.choice(frontier)]))

        # dominators = (objective[frontier] > objective[c]).all(axis=1)
        # remainders = (objective[frontier] >= objective[c]).any(axis=1)
        dominators = (objective[frontier] >= objective[c]).all(axis=1)
        remainders = (objective[frontier] > objective[c]).any(axis=1)

        if not dominators.any():
            frontier = np.append(frontier[remainders], [c])
            pioneer[c] = candidate[c]

        bests = ["%s: %s" % (obj_names[i], objective[:c+1, i].max()) for i in range(objective.shape[1])]
        print("%d  |  %d in frontier  |  bests: %s" % (c, frontier.size, ", ".join(bests)))
        
        if not dominators.any():
            with open(dump_file, "wb") as df: pk.dump((pioneer, objective, frontier), df)
            # with open(dump_file, "wb") as df: pk.dump((candidate, objective, frontier), df)
    
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

    cube_size = 2
    num_instances = 256
    tree_depth = 3
    max_depth = 1
    max_macros = 5
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 5
    wildcard_rate = .5
    rollout_length = 20
    num_candidates = 2**17
    # num_candidates = 64
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

    from candidate_set import CandidateSet
    candidate_set = CandidateSet(
        domain, bfs_tree, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length,
        num_patterns, num_instances, max_depth, max_macros)

    def evaluate_fun(candidate):
        candidate, objectives = candidate_set.evaluate(candidate)
        objectives = np.array(objectives)[1:]
        return candidate, objectives

    if dotrain:

        pareto_search(
            num_candidates,
            rng,
            spawn = candidate_set.spawn,
            mutate = candidate_set.mutate,
            evaluate = evaluate_fun,
            obj_names = obj_names,
            dump_file = dump_file,
        )

    import matplotlib.pyplot as pt

    if showresults:

        with open(dump_file, "rb") as f: (candidate, objectives, frontier) = pk.load(f)

        C = max(candidate.keys()) + 1
        objectives = objectives[:C]
        color = np.tile(np.linspace(.9, .5, C), (3,1)).T
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
        
        pt.figure(figsize=(15,5))
        pt.subplot(1,3,1)
        pt.scatter(*rando.T, color=color)
        # pt.scatter(*rando[frontier].T, color=color[frontier])

        pt.xlabel("- macro size")
        pt.ylabel("# godly solves")
        
        pt.subplot(1,3,2)
        pt.scatter(sorted(candidate.keys()), [candidate[c].match_counts.sum() for c in sorted(candidate.keys())], color='k')
        pt.scatter(frontier, [candidate[c].match_counts.sum() for c in frontier], color='r')
        pt.xlabel("candidate")
        pt.ylabel("total match count")
        pt.legend(["all pioneers", "frontier"])

        pt.subplot(1,3,3)
        # idx = np.argsort(objectives[frontier, 1])
        # pt.plot(objectives[frontier[idx], 1], sorted([candidate[c].match_counts.sum() for c in frontier[idx]]), '-ob')
        pt.scatter(objectives[frontier, 1], [candidate[c].match_counts.sum() for c in frontier])
        pt.xlabel("godly solves in frontier")
        pt.ylabel("total match count")

        pt.show()

    if postmortem:
        with open(dump_file, "rb") as f: (candidate, objectives, frontier) = pk.load(f)
        
        # # match counts for most godly solves
        # f = frontier[np.argmax(objectives[frontier,1])]
        # cand, objs = evaluate_fun(candidate[f])
        # pt.bar(np.arange(len(cand.match_counts)), cand.match_counts, width=.5, label="most godly")

        # # match counts for least godly solves
        # f = frontier[np.argmin(objectives[frontier,1])]
        # cand, objs = evaluate_fun(candidate[f])
        # pt.bar(np.arange(len(cand.match_counts))+.5, cand.match_counts, width=.5, label="least godly")

        # pt.legend()
        # pt.show()

        # variability due to instance sample
        candidate_set.num_instances = 512
        num_samples = 30
        godlies = np.empty((2,num_samples))
        for rep in range(num_samples):
            print("rep %d of %d" % (rep, num_samples))
            for f,fun in enumerate([np.argmin, np.argmax]):
                cand = candidate[frontier[fun(objectives[frontier,1])]]
                cand, objs = candidate_set.evaluate(cand)
                godlies[f,rep] = objs[2]
        godlies /= candidate_set.num_instances
        print("stat, less, more godly:")
        print("avg", godlies.mean(axis=1))
        print("std", godlies.std(axis=1))
        pt.hist(godlies.T, label=["least godly pioneer","most godly pioneer"])
        pt.xlabel("Godly solve rate")
        pt.ylabel("Frequency")
        pt.title("Variability across instance samples")
        pt.legend()
        pt.show()

