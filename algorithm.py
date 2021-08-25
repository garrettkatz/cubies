import numpy as np

def macro_search(state, domain, bfs_tree, pattern_database, max_depth, color_neutral=True):
    # returns result = (actions, neutral index, macro, new_state)
    # or result = False if there is no path to a macro or solved state

    paths = bfs_tree.paths(up_to_depth=max_depth)
    descendents = bfs_tree.states_rooted_at(state, up_to_depth=max_depth)

    patterns = pattern_database.patterns
    wildcards = pattern_database.wildcard
    macros = pattern_database.macros
    
    for k in range(len(paths)):
        actions, descendent = paths[k], descendents[k] 
    
        # Empty macro if problem is solved in descendent state
        if domain.is_solved_in(descendent): return actions, 0, (), domain.solved_state()

        # Non-empty macro if state matches a database pattern
        if color_neutral:

            # recolorings = domain.color_neutral_to(descendent)
            # recolorings = recolorings[:, np.newaxis, :]
            # matches = ((recolorings == patterns) | wildcards).all(axis=2)
            # r, p = np.unravel_index(matches.argmax(), matches.shape)
            # if matches[r, p]:
            #     recolored, macro = recolorings[r,0], macros[p]
            #     return (actions, r, macro, domain.execute(macro, recolored))

            for r, recolored in enumerate(domain.color_neutral_to(descendent)):
                matched = pattern_database.query(recolored)
                if matched:
                    _, macro = pattern_database.result()
                    return (actions, r, macro, domain.execute(macro, recolored))
        else:
            matched = pattern_database.query(descendent)
            if matched:
                _, macro = pattern_database.result()
                return (actions, 0, macro, domain.execute(macro, descendent))

    # Failure if no path to macro found
    return False

def run(state, domain, bfs_tree, pattern_database, max_depth, max_actions, color_neutral=True):
    # returns solved, plan
    # solved: True if path to solved state was found, False otherwise
    # plan: [...,(actions, sym index, macro),...] a sequence of macro_search results
    
    # Form plan one macro at a time
    plan = []
    num_actions = 0
    while True:

        # Search for next macro
        result = macro_search(state, domain, bfs_tree, pattern_database, max_depth, color_neutral)
        
        # Return failure if none found
        if result is False: return False, plan
        
        # Or if max actions exceeded
        actions, sym, macro, new_state = result
        num_actions += max(len(actions) + len(macro), 1) # make sure count always increases
        if num_actions > max_actions: return False, plan
        
        # Otherwise, execute search result
        plan.append((actions, sym, macro))
        state = new_state

        # Terminate once solved
        if domain.is_solved_in(state): return True, plan

# def macro_search(state, domain, bfs_tree, pattern_database, max_depth, neutral=True):
#     # returns result = (actions, symmetry index, macro)
#     # or result = False if there is no path to a macro or solved state
    
#     for actions, descendent in bfs_tree.rooted_at(state, up_to_depth=max_depth):
        
#         # Empty macro if problem is solved in descendent state
#         if neutral and domain.is_solved_in(descendent):
#             sym = (domain.orientations_of(descendent) == domain.solved_state()).all(axis=1).argmax()
#             return actions, sym, []
#         if not neutral and (domain.solved_state() == descendent).all():
#             sym = (domain.orientations_of(descendent) == domain.solved_state()).all(axis=1).argmax()
#             return actions, sym, []

#         # Non-empty macro if state matches a database pattern
#         matched = pattern_database.query(descendent)
#         if matched: return (actions,) + pattern_database.result()

#     # Failure if no path to macro found
#     return False

# def run(state, domain, bfs_tree, pattern_database, max_depth, max_actions, neutral=True):
#     # returns solved, plan
#     # solved: True if path to solved state was found, False otherwise
#     # plan: [...,(actions, sym index, macro),...] a sequence of macro_search results
    
#     # Form plan one macro at a time
#     plan = []
#     num_actions = 0
#     while True:

#         # Search for next macro
#         result = macro_search(state, domain, bfs_tree, pattern_database, max_depth, neutral)
        
#         # Return failure if none found
#         if result is False: return False, plan
        
#         # Or if max actions exceeded
#         actions, sym, macro = result
#         num_actions += max(len(actions) + len(macro), 1) # make sure count always increases
#         if num_actions > max_actions: return False, plan
        
#         # Otherwise, execute search result
#         for action in actions: state = domain.perform(action, state)
#         state = domain.orientations_of(state)[sym]
#         for action in macro: state = domain.perform(action, state)
#         plan.append(result)

#         # Terminate once solved
#         if neutral and domain.is_solved_in(state): return True, plan
#         if not neutral and (domain.solved_state() == state).all(): return True, plan

if __name__ == "__main__":

    import numpy as np

    # #### test macro_search
    # max_depth = 2

    # from cube import CubeDomain
    # domain = CubeDomain(3)

    # from tree import SearchTree
    # bfs_tree = SearchTree(domain, max_depth)

    # from pattern_database import PatternDatabase
    # # patterns = domain.solved_state().reshape(1,-1)
    # # macros = [((0,1,0),)]
    # patterns = domain.perform((0,1,1), domain.solved_state()).reshape(1,-1)
    # wildcards = np.zeros(patterns.shape, dtype=bool)
    # macros = [((0,1,3),)]
    # pattern_database = PatternDatabase(patterns, wildcards, macros, domain)

    # state = domain.solved_state()
    # actions = ((0,1,1),)
    # state = domain.execute(domain.reverse(macros[0]), state)
    # new_state = domain.color_neutral_to(state)[2,:]
    # invsym = (domain.color_neutral_to(new_state) == state).all(axis=1).argmax()    
    # state = domain.execute(domain.reverse(actions), new_state)
    # result = macro_search(state, domain, bfs_tree, pattern_database, max_depth)
    
    # print(result)
    # assert result
    # path, s, macro, new_state = result
    # print(path)
    # assert path == actions
    # print(s, invsym)
    # assert s == invsym
    # print(macro)
    # assert macro == macros[0]
    # print(new_state)
    # assert (new_state == domain.solved_state()).all()

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
        domain.execute([(0,1,1),(1,1,1)], state),
        domain.execute([(0,1,1),(1,1,1),(2,1,1),(1,1,1),(0,1,1)], state),
    ))
    wildcard = np.zeros(patterns.shape, dtype=bool)
    macros = (
        ((1,1,3),(0,1,3)),
        ((0,1,3),(1,1,3),(2,1,3)),
    )
    pattern_database = PatternDatabase(patterns, wildcard, macros, domain)

    matched = pattern_database.query(patterns[1])
    assert matched
    _, macro = pattern_database.result()
    assert macro == macros[1]

    matched = pattern_database.query(domain.solved_state())
    assert not matched

    actions = ((1,1,1),)
    state = domain.color_neutral_to(patterns[1])[4]
    invsym = (domain.color_neutral_to(state) == patterns[1]).all(axis=1).argmax()
    state = domain.execute(domain.reverse(actions), state)
    solved, plan = run(state, domain, bfs_tree, pattern_database, max_depth=1, max_actions=20)
    
    assert solved
    path, s, macro = plan[0]
    assert path == actions
    assert s == invsym
    assert macro == macros[1]

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
        state = domain.color_neutral_to(state)[sym]
        draw(state, str(sym), i)
        i += 1
        for action in macro:
            state = domain.perform(action, state)
            draw(state, str(action), i)
            i += 1

    pt.show()

