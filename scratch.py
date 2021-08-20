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
    # interesting discovery: matched macros can accidentally reverse the work of macro_search, causing infinite loop
    # results for unoptimized, simplistic random but correct method:
    # depth 3 roughly halves the tree (9k states, 5k rules)

    tree_depth = 4
    max_depth = 1
    cube_size = 2

    import numpy as np

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    # init = domain.random_state(20, np.random.default_rng())
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    paths, states = zip(*tree.rooted_at(init))
    states = np.array(states)

    unruly = set([s for s in range(len(states)) if len(paths[s]) > 1])
    patterns = []
    wildcards = []
    macros = []

    def macro_match(macro, path):
        # if len(macro) <= len(path) and macro == path[-len(macro):]: return True
        # if len(macro) < len(path) and macro == path[-len(macro)-1:-1]: return True
        # return False
        for hi in range(len(path)-max_depth, len(path)+1):
            if macro == path[hi-len(macro):hi]: return True
        return False

    while len(unruly) > 0:
        s = unruly.pop()
        print("|unruly| = %d (%d states total), |rules| = %d" % (len(unruly), len(states), len(patterns)))

        pattern = states[s]
        wildcard = np.zeros(len(pattern), dtype=bool)

        # macro length should be strictly greater than max_depth to reduce chance of loops
        ln = np.random.randint(max_depth+1, len(paths[s])+1)
        hi = np.random.randint(max(ln, len(paths[s])-max_depth), len(paths[s])+1)
        lo = hi - ln
        macro = paths[s][lo:hi]

        grounded = np.random.permutation(range(domain.state_size()))
        for i in grounded:
            wildcard[i] = True
            matched = ((states == pattern) | wildcard).all(axis=1)
            for m in np.flatnonzero(matched):
                if not macro_match(macro, paths[m]):
                    wildcard[i] = False
                    break

        matched = ((states == pattern) | wildcard).all(axis=1)
        unruly -= set(np.flatnonzero(matched))

        patterns.append(pattern)
        wildcards.append(wildcard)
        macros.append(domain.reverse(macro))

    patterns = np.array(patterns)
    wildcards = np.array(wildcards)

    print(len(patterns))
    print(len(macros))
    print(wildcards.sum() / wildcards.size)

    # confirm correctness
    from pattern_database import PatternDatabase
    pdb = PatternDatabase(patterns, wildcards, macros, domain, orientation_neutral=False)

    from algorithm import run
    import matplotlib.pyplot as pt

    for p, (path, prob_state) in enumerate(tree.rooted_at(init)):
        if len(path) == tree_depth: continue # otherwise macro_search could exit set where pdb is correct
        # print("checking %d" % (p))
        solved, plan = run(prob_state, domain, tree, pdb, max_depth, max_actions=10)
        state = prob_state
        for (actions, sym, macro) in plan:
            state = domain.execute(actions, state)
            state = domain.orientations_of(state)[sym]
            state = domain.execute(macro, state)

        if not domain.is_solved_in(state):

            numcols = 10
            numrows = 5
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

