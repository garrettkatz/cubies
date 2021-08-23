import pickle as pk
import itertools as it
import numpy as np
from pattern_database import PatternDatabase
from algorithm import run
from utils import softmax

class Candidate:
    def __init__(self, patterns, wildcards, macros, domain, orientation_neutral):
        self.patterns = patterns
        self.wildcards = wildcards
        self.macros = macros
        self.pdb = PatternDatabase(patterns, wildcards, macros, domain, orientation_neutral)
        self.godliness_sum = 0
        self.evaluation_count = 0

def check_domination_of(x, by):
    # assumes one or both arguments are 2d
    return (by >= x).all(axis=1) & (by > x).any(axis=1)

def screen(cand, state, path, rng, tree, max_depth, tree_depth, use_safe_depth):

    patterns = cand.patterns
    wildcards = cand.wildcards
    macros = cand.macros

    upgraded = False # becomes True if candidate gets upgraded

    # restrict any rules needed so that state will not trigger bad macros
    wr, wc = [], [] # rows and columns where wildcards are disabled
    triggered = ((state == patterns) | wildcards).all(axis=1)
    for r in np.flatnonzero(triggered):
        goodmacro = (len(macros[r]) <= len(path)) and macros[r] == path[:len(macros[r])]
        if not goodmacro:
            wr.append(r)
            wc.append(rng.choice(np.flatnonzero(state != patterns[r])))
    if len(wr) > 0:
        wildcards = wildcards.copy()
        wildcards[wr, wc] = False
        upgraded = True

    # check if state is in neighborhood of a trigger
    # due to incomplete tree it must also be triggered within distance to tree_depth
    # otherwise macro_search could exit set where pdb is correct
    safe_depth = max_depth
    if use_safe_depth: safe_depth = min(max_depth, tree_depth - len(path))
    triggered = False # until proven otherwise
    for _, neighbor in tree.rooted_at(state, up_to_depth=safe_depth):
        triggered = ((neighbor == patterns) | wildcards).all(axis=1).any()
        if triggered: break

    # if not, create a new rule triggered by state
    if not triggered:
        upgraded = True

        # if this code is reached, path is longer than max depth
        macro = path[:rng.integers(max_depth, len(path))+1] # random macro    
        pattern = state
        # wildcard = np.ones(pattern.shape, dtype=bool) # start with all wildcards which will gradually be disabled
        wildcard = (np.random.rand(*pattern.shape) < (len(path) / domain.god_number())) # more wildcards in deeper states

        # add to pdb
        patterns = np.append(patterns, pattern[np.newaxis,:], axis=0)
        wildcards = np.append(wildcards, wildcard[np.newaxis,:], axis=0)
        macros = macros + [macro]

    if upgraded:
        return patterns, wildcards, macros
    else:
        return None

if __name__ == "__main__":

    # config
    do_search = True
    show_results = True
    post_mortem = False

    # do_search = False
    # show_results = True
    # post_mortem = False

    # larger exploration is important for larger state spaces, at least with uniform state sampling
    # larger state spaces need a few rules to start getting any godliness
    # otherwise the initial candidate dominates its offspring and keeps getting selected
    # tree_depth = 11
    # use_safe_depth = False
    tree_depth = 8
    use_safe_depth = True
    exploration = 1
    state_sampling = "bfs order"
    # state_sampling = "uniform"
    # state_sampling = "soft depth"

    max_depth = 1
    cube_size = 2
    # valid_actions = None
    valid_actions = it.product((0,1,2), (0,), (0, 1, 2, 3)) # only spinning one plane on each axis for 2cube
    max_actions = 30
    orientation_neutral=False
    
    selection_policy = "hard ucb"
    # selection_policy = "soft ucb"
    # selection_policy = "uniform"

    obj_names = ["godliness", "folkliness"]

    num_search_iters = 2**13
    # candidate_buffer_size = num_search_iters
    candidate_buffer_size = 1024
    num_reps = 1
    # break_seconds = 30 * 60
    break_seconds = 0
    dump_dir = "psearch"

    animate_tree = False

    verbose = True

    # Set up domain and state-space
    from cube import CubeDomain
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth, orientation_neutral)
    paths, states = zip(*tree.rooted_at(init))
    states = np.array(states)
    paths = list(map(tuple, map(domain.reverse, paths))) # from state to solved
    dists = np.array(list(map(len, paths)))

    # random number generation
    rng = np.random.default_rng()

    def evaluate(cand, state, path):
        solved, plan = run(state, domain, tree, cand.pdb, max_depth, max_actions, orientation_neutral)
        if solved:
            soln_len = sum([len(actions) + len(macro) for (actions, _, macro) in plan])
            godliness = len(path) / max(1, soln_len)
            cand.godliness_sum += godliness
        else:
            cand.godliness_sum += 0
        cand.evaluation_count += 1
        
        godliness = cand.godliness_sum / cand.evaluation_count
        folkliness = -len(cand.macros) # larger pdb is worse

        return godliness, folkliness

    if do_search:

        from time import sleep
        for rep in range(num_reps):

            # set up data dump
            dump_file = "%s/rep_%d.pkl" % (dump_dir, rep)

            # set up candidate pool
            candidate = {} # saves the best candidates found so far
            selection_count = np.zeros(num_search_iters, dtype=int)
            objective = np.empty((num_search_iters, len(obj_names)))
            parent = -np.ones(num_search_iters, dtype=int)
            ranking = np.empty(num_search_iters, dtype=int) # rank = number of candidates that dominate (frontier is rank 0)
            state_counter = np.zeros(num_search_iters, dtype=int) # for ordered state sampling
        
            # initialize candidate with one rule for solved state
            candidate[0] = Candidate(
                patterns = states[:1,:].copy(),
                wildcards = np.zeros((1, domain.state_size()), dtype=bool),
                macros = [()],
                domain = domain,
                orientation_neutral = orientation_neutral)
            # maintain invariant that every candidate is ranked and evaluated at least once
            godliness, folkliness = evaluate(candidate[0], domain.solved_state(), ())
            objective[0] = godliness, folkliness
            ranking[0] = 0 # no other candidates yet
            num_cand = 1

            for n in range(1, num_search_iters):

                # enumerate currently buffered candidates
                keys = list(candidate.keys())
                
                # upper confidence bounds
                N = selection_count[keys]
                Q = -ranking[keys].copy() # ranking closer to 0 is better
                ucb_logits = Q + exploration * np.sqrt(np.log(n) / (N+1))

                ## select a candidate still saved in memory
                if selection_policy == "hard ucb": c = keys[ucb_logits.argmax()]
                if selection_policy == "soft ucb": c = rng.choice(keys, p = softmax(ucb_logits))
                if selection_policy == "uniform": c = rng.choice(keys)

                selection_count[c] += 1

                # check whether each neighbor is dominated by current candidate before evaluation
                was_dominated = check_domination_of(objective[:num_cand], by=objective[c])

                # sample a state
                if state_sampling == "uniform": s = rng.choice(len(states))
                if state_sampling == "soft depth": s = rng.choice(len(states), p = softmax(-dists))
                if state_sampling == "bfs order": s = state_counter[c] % len(states)
                state_counter[c] += 1
                state, path = states[s], paths[s]

                # evaluate and update objectives
                godliness, folkliness = evaluate(candidate[c], state, path)
                objective[c] = godliness, folkliness

                # update dominated status of neighbors after evaluation
                is_dominated = check_domination_of(objective[:num_cand], by=objective[c])

                # update rankings
                ranking[:num_cand] += (is_dominated.astype(int) - was_dominated.astype(int))
                ranking[c] = check_domination_of(objective[c], by=objective[:num_cand]).sum()

                # upgrade selected candidate if needed
                upgrade = screen(candidate[c], state, path, rng, tree, max_depth, tree_depth, use_safe_depth)
                if upgrade is not None:

                    # update candidate set
                    patterns, wildcards, macros = upgrade
                    candidate[num_cand] = Candidate(patterns, wildcards, macros, domain, orientation_neutral)
                    parent[num_cand] = c
                    selection_count[num_cand] = 0
                    state_counter[num_cand] = state_counter[c]

                    # carry forward parent metrics
                    candidate[num_cand].evaluation_count = candidate[c].evaluation_count
                    candidate[num_cand].godliness_sum = candidate[c].godliness_sum
                    godliness = candidate[num_cand].godliness_sum / candidate[num_cand].evaluation_count
                    folkliness = -len(candidate[num_cand].macros)
                    objective[num_cand] = godliness, folkliness

                    # update rankings
                    ranking[:num_cand] += check_domination_of(objective[:num_cand], by = objective[num_cand])
                    ranking[num_cand] = check_domination_of(objective[num_cand], by = objective[:num_cand]).sum()

                    # update num candidates
                    num_cand += 1

                    # discard most dominated candidate if buffer size reached
                    if len(candidate) > candidate_buffer_size:
                        worst = keys[ranking[keys].argmax()]
                        candidate.pop(worst)

                # save progress
                progress = selection_count[:num_cand], parent[:num_cand], objective[:num_cand], ranking[:num_cand]
                with open(dump_file, "wb") as df: pk.dump(progress, df)

                # if verbose and n % (10**int(np.log10(n))) == 0:
                if verbose:

                    # bests = ["%s: %s" % (obj_names[i], objective[(selection_count > 0), i].max()) for i in range(objective.shape[1])]
                    bests = ["%s: %s" % (obj_names[i], objective[:num_cand, i].max()) for i in range(objective.shape[1])]
                    print("%d/%d: selected <= %d | counter <= %d | bests: %s"  % (n, num_search_iters, selection_count.max(), state_counter.max(), ", ".join(bests)))
                    # print("%d | %d in frontier | %d spawns | counts <=%d | bests: %s" % (c, frontier.size, num_spawns, count[:c+1].max(), ", ".join(bests)))
                    # print("iter %d: %d <= %d rules, %f wildcard, done=%s (k=%d)" % (epoch, len(macros), len(states), wildcards.sum() / wildcards.size, done, k))

            print("Breaking for %s seconds..." % str(break_seconds))
            sleep(break_seconds)

    if show_results:

        import matplotlib.pyplot as pt

        # load progress
        rep_results = []
        for rep in range(num_reps):
            dump_file = "%s/rep_%d.pkl" % (dump_dir, rep)
            with open(dump_file, "rb") as df: rep_results.append(pk.load(df))

        selection_count, parent, objective, ranking = rep_results[0]
        nc = len(ranking)
        # elites = np.flatnonzero(ranking[1:] < 20) + 1
        elites = np.arange(1, nc)
        if animate_tree:
            pt.ion()
            pt.figure()
        else:
            pt.figure(figsize=(20,15))
        pt.xlabel("folkliness")
        pt.ylabel("godliness")
        # for c in range(1,nc):
        for c in elites:
            pt.plot(
                [objective[c,obj_names.index("folkliness")], objective[parent[c],obj_names.index("folkliness")]],
                [objective[c,obj_names.index("godliness")], objective[parent[c],obj_names.index("godliness")]],
                '-ko')
            if animate_tree: pt.pause(0.01)
        if not animate_tree: pt.show()

        # pt.figure(figsize=(15, 5))
        # for rep, results in enumerate(rep_results):
        #     selection_count, parent, objective, ranking = results
        #     # sc = np.flatnonzero(selection_count > 0)
        #     sc = np.arange(len(ranking))
        #     nc = len(ranking)

        #     num_plots = 5

        #     pt.subplot(1, num_plots, 1)
        #     pt.bar(np.arange(len(sc)), objective[sc, obj_names.index("godliness")], label=str(rep))
        #     pt.xlabel("candidate")
        #     pt.ylabel("godliness")

        #     pt.subplot(1, num_plots, 2)
        #     pt.bar(np.arange(nc), selection_count, label=str(rep))
        #     pt.xlabel("candidate")
        #     pt.ylabel("selection count")

        #     pt.subplot(1, num_plots, 3)
        #     pt.scatter(selection_count[sc], ranking[sc], label=str(rep))
        #     pt.xlabel("selection count")
        #     pt.ylabel("ranking")

        #     pt.subplot(1, num_plots, 4)
        #     pt.scatter(selection_count[sc], objective[sc, obj_names.index("godliness")], label=str(rep))
        #     pt.xlabel("selection count")
        #     pt.ylabel("godliness")

        #     pt.subplot(1, num_plots, 5)
        #     pt.scatter(objective[sc, obj_names.index("folkliness")], objective[sc, obj_names.index("godliness")], label=str(rep))
        #     pt.xlabel("folkliness")
        #     pt.ylabel("godliness")

        # pt.legend()
        # pt.show()
