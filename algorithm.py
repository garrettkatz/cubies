def macro_search(state, domain, bfs_tree, pattern_database, max_depth):
    # returns result = (actions, symmetry index, macro)
    # or result = False if there is no path to a macro or solved state
    
    for actions, permutation in bfs_tree:

        # Don't exceed maximum search depth
        if len(actions) > max_depth: continue    

        # Compute descendent state
        descendent = state[permutation]
        
        # Empty macro if problem is solved in descendent state
        if domain.is_solved_in(descendent):
            sym = (domain.symmetries_of(descendent) == domain.solved_state()).all(axis=1).argmax()
            return actions, sym, []

        # Non-empty macro if state matches a database pattern
        matched = pattern_database.query(descendent)
        if matched: return (actions,) + pattern_database.result()

    # Failure if no path to macro found
    return False

def run(state, domain, bfs_tree, pattern_database, max_depth, max_macros):
    # returns solved, plan
    # solved: True if path to solved state was found, False otherwise
    # plan: [...,(actions, sym index, macro),...] a sequence of macro_search results
    
    # Form plan one macro at a time
    plan = []
    for num_macros in range(max_macros):

        # Search for next macro
        result = macro_search(state, domain, bfs_tree, pattern_database, max_depth)
        
        # Return failure if none found
        if result is False: return False, plan
        
        # Otherwise, execute search result
        actions, sym, macro = result
        for action in actions: state = domain.perform(action, state)
        state = domain.symmetries_of(state)[sym]
        for action in macro: state = domain.perform(action, state)
        plan.append(result)

        # Terminate once solved
        if domain.is_solved_in(state): return True, plan
    
    # At this point, no successful plan was found
    return False, plan

if __name__ == "__main__":

    # #### test macro_search
    # max_depth = 2

    # from cube import CubeDomain
    # domain = CubeDomain(3)

    # from tree import SearchTree
    # bfs_tree = SearchTree(domain, max_depth)

    # from pattern_database import PatternDatabase
    # # patterns = domain.solved_state().reshape(1,-1)
    # # macros = [((0,0,0),)]
    # patterns = domain.perform((0,0,1), domain.solved_state()).reshape(1,-1)
    # macros = [((0,0,3),)]
    # pattern_database = PatternDatabase(patterns, macros)

    # state = domain.solved_state()
    # state = domain.perform((0,0,1),state)
    # state = domain.symmetries_of(state)[2,:]
    # state = domain.perform((1,0,1),state)
    # result = macro_search(state, domain, bfs_tree, pattern_database, max_depth)
    # if result != False:
    #     path, s, macro = result
    #     print(path)
    #     print(s)
    #     print(macro)

    #### test run
    max_depth = 2

    from cube import CubeDomain
    domain = CubeDomain(3)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    import numpy as np
    from pattern_database import PatternDatabase
    state = domain.solved_state()
    patterns = np.stack((
        domain.execute([(0,0,1),(1,0,1)], state),
        domain.execute([(0,0,1),(1,0,1),(2,0,1),(1,0,1),(0,0,1)], state),
    ))
    wildcard = np.zeros(patterns.shape, dtype=bool)
    macros = (
        [(1,0,3),(0,0,3)],
        [(0,0,3),(1,0,3),(2,0,3)],
    )
    pattern_database = PatternDatabase(patterns, wildcard, macros, domain)

    matched = pattern_database.query(domain.symmetries_of(patterns[1])[15])
    print(matched)
    if matched:
        sym, macro = pattern_database.result()
        print(sym, macro)

    # state = domain.perform((1,0,1), patterns[1])
    state = domain.perform((1,0,1), domain.symmetries_of(patterns[1])[21])
    # state = domain.symmetries_of(patterns[1])[15]
    # state = patterns[1]
    solved, plan = run(state, domain, bfs_tree, pattern_database, max_depth=1, max_macros=2)
    if solved:

        import matplotlib.pyplot as pt
        def draw(st, title, i):
            ax = pt.subplot(4, 6, i)
            domain.render(st, ax, 0, 0)
            ax.axis("equal")
            ax.axis('off')
            ax.set_title(title)

        i = 1
        draw(state, "initial", i)
        i += 1
        for (actions, sym, macro) in plan:
            print(actions)
            print(sym)
            print(macro)

            for action in actions:
                state = domain.perform(action, state)
                draw(state, str(action), i)
                i += 1
            state = domain.symmetries_of(state)[sym]
            draw(state, str(sym), i)
            i += 1
            for action in macro:
                state = domain.perform(action, state)
                draw(state, str(action), i)
                i += 1

        pt.show()

