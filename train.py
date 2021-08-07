import numpy as np
import pickle as pk
from utils import softmax

def pareto_chains(num_candidates, rng, spawn, mutate, evaluate, obj_names, dump_file):

    candidate = {}
    objective = np.empty((num_candidates, len(obj_names)))
    candidate[0], objective[0] = evaluate(spawn())

    parent = {} # ancestry pointers
    frontier = {0} # candidates not currently dominated by a child or parent
    pioneers = dict(candidate) # candidates that were ever in a frontier

    num_children = np.zeros(num_candidates)
    num_nondominated_children = np.zeros(num_candidates)

    for c in range(1, num_candidates):

        # sample a candidate parent for mutation
        p = rng.choice(list(frontier))

        # mutate and evaluate selection
        candidate[c], objective[c] = evaluate(mutate(candidate[p]))
        num_children[p] += 1

        # track whether frontier changed
        frontier_changed = False

        # add child to frontier as long as it is not dominated by parent
        parent_dominates = (objective[p] >= objective[c]).all() and (objective[p] > objective[c]).any()
        if not parent_dominates:
            frontier.add(c)
            frontier_changed = True
            pioneers[c] = candidate[c]
            parent[c] = p
            num_nondominated_children[p] += 1

        # discard all ancesters dominated by child
        while p > 0:
            child_dominates = (objective[c] >= objective[p]).all() and (objective[c] > objective[p]).any()
            if child_dominates:
                frontier.discard(p)
                frontier_changed = True
            p = parent[p]

        # save progress when frontier changes
        if frontier_changed:
            with open(dump_file, "wb") as df: pk.dump((pioneers, objective, frontier, num_children, num_nondominated_children), df)

        bests = ["%s: %s" % (obj_names[i], objective[:c+1, i].max()) for i in range(objective.shape[1])]
        print("%d  |  %d in frontier  |  bests: %s" % (c, len(frontier), ", ".join(bests)))

    return candidate, objective, frontier

def pareto_search(num_candidates, rng, spawn, mutate, evaluate, obj_names, dump_file):

    candidate = {}
    objective = np.empty((num_candidates, len(obj_names)))

    candidate[0], objective[0] = evaluate(spawn())
    frontier = np.array([0]) # set of all candidates that are not strictly dominated by others explored so far
    pioneer = dict(candidate) # candidates that were ever in a frontier
    concentration = {tuple(objective[0]): 1} # frequency of each point in objective space

    for c in range(1, num_candidates):

        # sample a candidate for mutation
        # s = rng.choice(frontier) # uniform
        s = rng.choice(frontier, # more likely if less concentrated
            p = softmax([-concentration[tuple(objective[f])] for f in frontier]))

        # mutate and evaluate selection
        candidate[c], objective[c] = evaluate(mutate(candidate[s]))
        
        # update concentration
        concentration[tuple(objective[c])] = concentration.get(tuple(objective[c]), 0) + 1

        # update frontier if child is not strictly dominated
        # x strictly dominates y if (x > y).all()
        if (objective[frontier] <= objective[c]).any(axis=1).all():

            # keep existing frontier members that are still not strictly dominated
            keep = (objective[frontier] >= objective[c]).any(axis=1)

            # update frontier and pioneers
            frontier = np.append(frontier[keep], [c])
            pioneer[c] = candidate[c]

            # save progress
            with open(dump_file, "wb") as df: pk.dump((pioneer, objective, frontier, concentration), df)

        bests = ["%s: %s" % (obj_names[i], objective[:c+1, i].max()) for i in range(objective.shape[1])]
        print("%d | %d in frontier | <=%d repeats | bests: %s" % (c, frontier.size, max(concentration.values()), ", ".join(bests)))

    return candidate, objective, frontier, concentration

if __name__ == "__main__":

    # dotrain = True
    # showresults = False
    # postmortem = False

    # dotrain = False
    # showresults = True
    # postmortem = False

    # dotrain = False
    # showresults = False
    # postmortem = True

    # dotrain = True
    # showresults = True
    # postmortem = False

    # dotrain = False
    # showresults = True
    # postmortem = True

    dotrain = True
    showresults = True
    postmortem = True

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
    # num_candidates = 1024
    obj_names = ["macro size", "godly solves"]
    # mutate = "mutate"
    # mutate = "mutate_scores"
    mutate = "mutate_macro"
    dump_file = "data.pkl"

    rng = np.random.default_rng()

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    solved = domain.solved_state()
    valid_actions = list(domain.valid_actions(solved))

    from tree import SearchTree
    bfs_tree = SearchTree(domain, tree_depth)

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
        # pareto_chains(
            num_candidates,
            rng,
            spawn = candidate_set.spawn,
            # mutate = candidate_set.mutate,
            mutate = getattr(candidate_set, mutate),
            evaluate = evaluate_fun,
            obj_names = obj_names,
            dump_file = dump_file,
        )

    import matplotlib.pyplot as pt

    if showresults:

        # dump_file = "data_2.pkl"
        with open(dump_file, "rb") as df: data = pk.load(df)

        (candidate, objectives, frontier) = data[:3]
        # frontier = np.array(sorted(frontier)) # pareto chains

        C = max(candidate.keys()) + 1
        objectives = objectives[:C]
        color = np.tile(np.linspace(.9, .5, C), (3,1)).T
        # # color = np.ones((C, 3))
        # # color[:,0] = np.linspace(0, .5, C)
        # # color[frontier,2] = color[frontier, 0]
        # # color[frontier,0] = 1
        # color = np.zeros((C, 3))
        # color[:,0] = 1
        # color[frontier,2] = color[frontier, 0]
        # color[frontier,0] = 0
        rando = (objectives + .0*(rng.random(objectives.shape) - .5))
        
        pt.figure(figsize=(15,5))
        pt.subplot(1,3,1)
        pt.scatter(*rando.T, color=color)
        pt.scatter(*rando[frontier].T, color='k')
        # pt.scatter(*rando.T, s=1, color=color)
        # pt.scatter(*rando[frontier].T, s=1, color='k')

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

        from candidate_set import Candidate

        # dump_file = "data_2.pkl"
        # dump_file = "data_2_saturated.pkl"
        with open(dump_file, "rb") as f:  data = pk.load(f)
        (candidate, objectives, frontier) = data[:3]
        pioneers = list(sorted(candidate.keys()))
        
        # # variability due to instance sample
        # # (least and most godly candidates should have significantly different godliness rates across instance samples)
        # candidate_set.num_instances = 256
        # num_samples = 30
        # godlies = np.empty((2,num_samples))
        # for rep in range(num_samples):
        #     print("rep %d of %d" % (rep, num_samples))
        #     for f,fun in enumerate([np.argmin, np.argmax]):
        #         # cand = candidate[frontier[fun(objectives[frontier,1])]]
        #         cand = candidate[pioneers[fun(objectives[pioneers,1])]]
        #         cand, objs = candidate_set.evaluate(Candidate(cand.patterns, cand.macros))
        #         godlies[f,rep] = objs[2]
        # godlies /= candidate_set.num_instances
        # print("stat, less, more godly:")
        # print("avg", godlies.mean(axis=1))
        # print("std", godlies.std(axis=1))
        # pt.hist(godlies.T, label=["least godly pioneer","most godly pioneer"])
        # pt.xlabel("Godly solve rate")
        # pt.ylabel("Frequency")
        # pt.title("Variability across instance samples")
        # pt.legend()
        # pt.show()

        # # match counts for each pattern in least/most godly solves
        # # expected higher for lower indices since patterns are matched first-to-last
        # # want all match counts > 0 before increasing num_patterns
        # for i, (fun, lab) in enumerate(zip([np.argmin, np.argmax], ["min", "max"])):
        #     f = frontier[fun(objectives[frontier,1])]
        #     cand = candidate[f]
        #     cand, objs = candidate_set.evaluate(Candidate(cand.patterns, cand.macros))
        #     pt.bar(np.arange(cand.match_counts.shape[1]) + i*.5, cand.match_counts.sum(axis=0), width=.5, label=lab + " godly")
        # pt.legend()
        # pt.xlabel("pattern index")
        # pt.ylabel("match counts")
        # pt.show()

        # # match count vs macro length (see if minimal match count selected for mutation also has length 1)
        # for i, (fun, lab) in enumerate(zip([np.argmin, np.argmax], ["min", "max"])):

        #     f = frontier[fun(objectives[frontier,1])]
        #     cand = candidate[f]
        #     cand, objs = candidate_set.evaluate(Candidate(cand.patterns, cand.macros))

        #     # scores = (cand.match_counts[cand.successes,:] - cand.match_counts[~cand.successes,:]).sum(axis=0)
        #     # p = np.argmin(scores)
        #     # print("%s godly mutate index %d: macro length = %d" % (lab, p, len(cand.macros[p])))

        #     rando_counts = cand.match_counts.sum(axis=0) + .25*(rng.random(cand.match_counts.shape[1]) - .5)
        #     rando_length = np.array(list(map(len, cand.macros))) + .25*(rng.random(len(cand.macros)) - .5)
        #     pt.scatter(rando_counts, rando_length, label=lab + " godly")
        # pt.xlabel("match count")
        # pt.ylabel("macro length")
        # pt.legend()
        # pt.show()

        # clarify sources of ungodliness
        candidate_set.num_instances = 512
        cand = candidate[frontier[np.argmax(objectives[frontier,1])]]
        # cand = candidate[rng.choice(frontier)]
        cand = Candidate(cand.patterns, cand.macros)
        cand, obj = candidate_set.evaluate(cand)

        # pt.bar(np.arange(cand.match_counts.shape[1]), cand.match_counts.sum(axis=0), width=.3, label="total")
        # pt.bar(np.arange(cand.match_counts.shape[1]) + .3, cand.match_counts[cand.successes,:].sum(axis=0), width=.3, label="good")
        # pt.bar(np.arange(cand.match_counts.shape[1]) + .6, cand.match_counts[~cand.successes,:].sum(axis=0), width=.3, label="fail")
        # pt.legend()
        # pt.xlabel("pattern index")
        # pt.ylabel("match counts")
        # pt.title("%d queries" % cand.num_queries.sum())
        # pt.show()
        
        # # histogram of macro counts across frontier
        # # frontier = frontier[:3]
        # for f in frontier:
        #     print(f)
        #     cand = candidate[f]
        #     candidate[f], _ = candidate_set.evaluate(Candidate(cand.patterns, cand.macros))
        # pt.hist([
        #     np.concatenate([candidate[f].macro_counts[candidate[f].successes] for f in frontier]),
        #     np.concatenate([candidate[f].macro_counts[~candidate[f].successes] for f in frontier])
        # ])
        # pt.legend(["pass","fail"])
        # pt.show()

        # how many macros executed on successful and failed runs?
        # how many actions executed ""? (plan length)
        pt.subplot(2,1,1)
        pt.bar(np.arange(len(cand.macro_counts)), cand.macro_counts * cand.successes, width=.3, label="pass")
        pt.bar(np.arange(len(cand.macro_counts)) + .3, cand.macro_counts * (1 - cand.successes), width=.3, label="fail")
        pt.xlabel("Instance")
        pt.ylabel("Macro counts")
        pt.legend()
        pt.subplot(2,1,2)
        pt.bar(np.arange(len(cand.action_counts)), cand.action_counts * cand.successes, width=.3, label="pass")
        pt.bar(np.arange(len(cand.action_counts)) + .3, cand.action_counts * (1 - cand.successes), width=.3, label="fail")
        pt.xlabel("Instance")
        pt.ylabel("Action counts")
        pt.legend()
        pt.show()
        
        # visualize most godly candidate
        candidate_set.num_instances = 512
        cand = candidate[frontier[np.argmax(objectives[frontier,1])]]
        cand = Candidate(cand.patterns, cand.macros)
        cand, obj = candidate_set.evaluate(cand)
        # candidate_set.show(cand)
        
        # just show best and worst patterns
        idx = np.argsort((cand.match_counts[cand.successes].sum(axis=0) - cand.match_counts[~cand.successes].sum(axis=0)))
        patterns = []
        macros = []
        for i in range(4):
            patterns.append(cand.patterns[idx[i]])
            macros.append(cand.macros[idx[i]])
        for i in range(4):
            patterns.append(cand.patterns[idx[i-4]])
            macros.append(cand.macros[idx[i-4]])
        candidate_set.show(Candidate(patterns, macros))

        # count frequency of each frontier point in objective space
        # freqs = {}
        # for f in frontier:
        #     obj = tuple(objectives[f])
        #     freqs[obj] = freqs.get(obj, 0) + 1
        concentration = data[3]
        freqs = concentration
        # freqs = {key: concentration[key] for key in [tuple(obj) for obj in objectives[frontier,:]]}

        pt.plot(sorted(freqs.values()))
        pt.title("Sorted frequencies of distinct objective vectors")
        pt.show()
        
        xs, ys = zip(*freqs.keys())
        cs = []
        for (x,y) in freqs.keys():
            cs.append((.1 + .8*(freqs[x,y] / max(freqs.values())),) * 3)
        pt.scatter(xs, ys, c=cs)
        pt.xlabel("-macro size")
        pt.ylabel("godly solves")
        pt.title("Objective vector frequencies (dark is less)")
        pt.show()
