if __name__ == "__main__":

    # #### check how many interstates of scramble are solvable by algorithm
    # from cube import CubeDomain
    # from pattern_database import PatternDatabase
    # domain = CubeDomain(2)
    # state = domain.solved_state()
    # patterns, wildcard, macros =  [],[],[]
    # pattern_database = PatternDatabase(patterns, wildcard, macros, domain)

    # from tree import SearchTree
    # bfs_tree = SearchTree(domain, max_depth=4)

    # import numpy as np
    # rng = np.random.default_rng()
    # valid_actions = tuple(domain.valid_actions(state))
    # states = []
    # actions = []
    # for s in range(20):
    #     action = rng.choice(valid_actions)
    #     state = domain.perform(action, state)
    #     states.append(state)
    #     actions.append(action)

    # from algorithm import run
    # num_action = []
    # was_solved = []
    # for state in states:
    #     solved, plan = run(state, domain, bfs_tree, pattern_database, max_depth=4, max_actions=10)
    #     was_solved.append(solved)
    #     num_action.append(sum([len(a)+len(m) for (a,_,m) in plan]))
    
    # for (a, s, n) in zip(actions, was_solved, num_action):
    #     print(a, n, s)


    #### test simplistic method to generate complete and correct macro set for tree
    # for every tree state a rule's trigger matches, its macro must match leading actions towards solved
    # no state should trigger more than one rule? and with max depth 0, every state should trigger at least one
    # interesting discovery: matched macros can accidentally reverse the work of macro_search, causing infinite loop
    # results for unoptimized, simplistic random but correct method:

    tree_depth = 3
    max_depth = 0
    cube_size = 3
    max_actions = 30

    import numpy as np

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    paths, states = zip(*tree.rooted_at(init))
    states = np.array(states)
    num_rulers = np.zeros(len(states))
    num_rulers[0] = 1 # consider the solved state ruled
    state_index = {states[s].tobytes(): s for s in range(len(states))}

    # unruly = set([s for s in range(len(states)) if len(paths[s]) > max_depth])
    patterns = []
    wildcards = []
    macros = []

    def macro_match(macro, path):
        # for hi in range(len(path)-max_depth, len(path)+1):
        #     if macro == path[hi-len(macro):hi]: return True
        # return False
        return macro == path[-len(macro):]

    # goal of algorithm: each state is within max_depth of triggering a rule, and no state triggers more than one rule

    # while len(unruly) > 0:
    #     s = unruly.pop()
    #     print("|unruly| = %d (%d states total), |rules| = %d" % (len(unruly), len(states), len(patterns)))
    for k,s in enumerate(range(len(states))):
    # for k,s in enumerate(np.random.permutation(range(len(states)))):
        print("state %d of %d, |rules| = %d" % (k, len(states), len(patterns)))


        # if state is within max_depth of tree_depth, skip it
        # otherwise macro_search could exit set where pdb is correct
        if len(paths[s]) + max_depth > tree_depth: continue

        # if state is within max_depth of triggering a rule, skip it
        for _, other_state in tree.rooted_at(states[s], up_to_depth=max_depth):
            if other_state.tobytes() in state_index:
                os = state_index[other_state.tobytes()]
                if num_rulers[os] > 0: break
        if num_rulers[os] > 0: continue

        # macro length should be strictly greater than max_depth to reduce chance of loops
        ln = np.random.randint(min(max_depth+1, len(paths[s])), len(paths[s])+1)
        # hi = np.random.randint(max(ln, len(paths[s])-max_depth), len(paths[s])+1)
        hi = len(paths[s]) # new macro starting from this state
        lo = hi - ln
        macro = paths[s][lo:hi]

        pattern = states[s]
        wildcard = np.zeros(len(pattern), dtype=bool)

        grounded = np.random.permutation(range(domain.state_size()))
        for i in grounded:
            wildcard[i] = True
            matched = ((states == pattern) | wildcard).all(axis=1)
            for m in np.flatnonzero(matched):
                if not macro_match(macro, paths[m]) or num_rulers[m] > 0:
                    wildcard[i] = False
                    break
        # for i in range(domain.state_size()):
        #     grounded = np.flatnonzero(~wildcard)
        #     matchcounts = np.zeros(len(grounded))
        #     for w,j in enumerate(grounded):
        #         wildcard[j] = True
        #         matched = ((states == pattern) | wildcard).all(axis=1)
        #         for m in np.flatnonzero(matched):
        #             if not macro_match(macro, paths[m]) or num_rulers[m] > 0:
        #                 wildcard[j] = False
        #                 break
        #         if wildcard[j]: matchcounts[w] = matched.sum()
        #         wildcard[j] = False
        #     if (matchcounts > 0).any():
        #         w = np.argmax(matchcounts)
        #         j = grounded[w]
        #         wildcard[j] = True
        #     else:
        #         break

        matched = ((states == pattern) | wildcard).all(axis=1)
        num_rulers[matched] += 1
        # unruly -= set(np.flatnonzero(matched))

        patterns.append(pattern)
        wildcards.append(wildcard)
        macros.append(domain.reverse(macro))

    patterns = np.array(patterns)
    wildcards = np.array(wildcards)

    print("wildcard percentage: ", wildcards.sum() / wildcards.size)
    num_rulers = num_rulers[1:] # exclude solved state
    print("min/max/mean times any state matches a rule:", num_rulers.min(), num_rulers.max(), num_rulers.mean())
    print("(max_depth = %d)" % max_depth)

    # confirm correctness
    from pattern_database import PatternDatabase
    pdb = PatternDatabase(patterns, wildcards, macros, domain, orientation_neutral=False)

    from algorithm import run
    import matplotlib.pyplot as pt

    num_checked = 0
    opt_moves = []
    alg_moves = []
    for p, (path, prob_state) in enumerate(tree.rooted_at(init)):
        if len(path) + max_depth > tree_depth: continue # otherwise macro_search could exit set where pdb is correct
        num_checked += 1
        # print("checked %d" % (num_checked))
        solved, plan = run(prob_state, domain, tree, pdb, max_depth, max_actions)
        state = prob_state
        for (actions, sym, macro) in plan:
            state = domain.execute(actions, state)
            state = domain.orientations_of(state)[sym]
            state = domain.execute(macro, state)

        if len(path) > 0:
            opt_moves.append(len(path))
            alg_moves.append(sum([len(a)+len(m) for a,_,m in plan]))

        if not domain.is_solved_in(state):

            numcols = 20
            numrows = 6
            state = prob_state
            domain.render_subplot(numrows,numcols,1,state)
            for a, action in enumerate(domain.reverse(path)):
                state = domain.perform(action, state)
                domain.render_subplot(numrows,numcols,a+2,state)

            state = prob_state
            domain.render_subplot(numrows,numcols,numcols+1,state)
            sp = numcols + 2
            for (actions, sym, macro) in plan:
                for a,action in enumerate(actions):
                    state = domain.perform(action, state)
                    ax = domain.render_subplot(numrows,numcols, sp, state)
                    if a == 0:
                        ax.set_title("|" + str(action))
                    else:
                        ax.set_title(str(action))
                    sp += 1
                state = domain.orientations_of(state)[sym]
                ax = domain.render_subplot(numrows,numcols, sp, state)
                ax.set_title(str(sym))
                sp += 1
                for action in macro:
                    state = domain.perform(action, state)
                    ax = domain.render_subplot(numrows,numcols, sp, state)
                    ax.set_title(str(action))
                    sp += 1

            pt.show()

        assert solved

    alg_moves = np.array(alg_moves)
    opt_moves = np.array(opt_moves)
    alg_opt = alg_moves / opt_moves
    print("alg/opt min,max,mean", (alg_opt.min(), alg_opt.max(), alg_opt.mean()))
    print("alg min,max,mean", (alg_moves.min(), alg_moves.max(), alg_moves.mean()))
    print("Solved %d (%d/%d = %f)" % (num_checked, len(patterns), num_checked, len(patterns)/num_checked))

