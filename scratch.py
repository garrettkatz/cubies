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


    #### test simplistic method to generate correct macro set for tree
    # interesting discovery: matched macros can accidentally reverse the work of macro_search, causing infinite loop
    # conditions for correctness:
    #   every state within max_depth of triggering a rule or solving the cube
    #   every macro has length strictly greater than max_depth
    #   every macro is leading portion of optimal path to solved for every state that triggers it
    # factors influencing optimality:
    #   macro length should tend to be substantially larger than max_depth

    # config
    tree_depth = 3
    max_depth = 0
    cube_size = 2
    max_actions = 30
    orientation_neutral=False

    import numpy as np

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth, orientation_neutral)
    paths, states = zip(*tree.rooted_at(init))
    states = np.array(states)
    state_index = {states[s].tobytes(): s for s in range(len(states))}
    num_rulers = np.zeros(len(states))
    num_rulers[0] = 1 # solved state is ruled

    patterns = []
    wildcards = []
    macros = []
    
    # pseudocode:
    # while some states are unprocessed:
    #     nondet choose next state for processing
    #     if state within max_depth of triggering a rule or solve, continue
    #     nondet choose macro = leading portion of opt path from state to solve, length strictly greater than maxdepth
    #     classify all states into those whose leading optimal paths match macro (A), and those who don't (B)
    #     nondet choose wildcard possibility that triggers current state and none of (B)
    #     add state, wildcard, and macro to ruleset
    
    unprocessed = set(np.random.permutation(range(len(states))))
    while len(unprocessed) > 0:

        print("%d of %d states done, |rules| = %d" % (len(states) - len(unprocessed), len(states), len(patterns)))

        # nondet choose next state for processing
        s = unprocessed.pop()
        pattern = states[s]

        # if state within max_depth of triggering a rule or solve, continue
        # due to incomplete tree it must also be ruled within distance to tree_depth
        # otherwise macro_search could exit set where pdb is correct
        near_rule = False
        rule_depth = min(max_depth, tree_depth - len(paths[s]))
        for path, neighbor in tree.rooted_at(pattern, up_to_depth=rule_depth):
            if orientation_neutral:
                for oriented in domain.orientations_of(neighbor):
                    if oriented.tobytes() in state_index:
                        near_rule = (num_rulers[state_index[oriented.tobytes()]] > 0)
                        if near_rule: break
            # elif neighbor.tobytes() in state_index:
            else:
                near_rule = (num_rulers[state_index[neighbor.tobytes()]] > 0)
            if near_rule: break
        if near_rule: continue

        # nondet choose macro = leading portion of opt path from state, len(macro) > max_depth
        lo = np.random.randint(len(paths[s]) - max_depth)
        macro = paths[s][lo:]
        # macro = paths[s]

        # classify all states into those whose leading optimal paths match macro, and those who don't
        matched_macro = np.zeros(len(states), dtype=bool)
        for m in range(len(states)):
            matched_macro[m] = (len(paths[m]) >= len(macro)) and (macro == paths[m][-len(macro):])

        # nondet choose wildcard possibility that triggers current state and none whose optimal paths don't match
        wildcard = np.zeros(len(pattern), dtype=bool)
        ordering = np.random.permutation(range(domain.state_size()))
        for i in ordering:
            wildcard[i] = True
            triggered = ((states == pattern) | wildcard).all(axis=1)
            if (triggered & ~matched_macro).any(): wildcard[i] = False

        # updated ruled status
        matched = ((states == pattern) | wildcard).all(axis=1)
        num_rulers[matched] += 1

        # add state, wildcard, and macro to ruleset
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
    pdb = PatternDatabase(patterns, wildcards, macros, domain, orientation_neutral)

    from algorithm import run
    import matplotlib.pyplot as pt

    num_checked = 0
    opt_moves = []
    alg_moves = []
    for p, (path, prob_state) in enumerate(tree.rooted_at(init)):
        # if len(path) + max_depth > tree_depth: continue # otherwise macro_search could exit set where pdb is correct
        num_checked += 1
        if p % (10**int(np.log10(p+1))) == 0: print("checked %d" % (num_checked))

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

