def macro_search(state, domain, bfs_tree, pattern_database, max_depth):
    # returns result = (actions, symmetry index, macro)
    # or result = False if there is no path to a macro
    
    for actions, permutation in bfs_tree:

        # Don't exceed maximum search depth
        if len(actions) > max_depth: continue    

        # Compute descendent state
        descendent = state[permutation]
        
        # Consider all symmetric states
        sym_states = domain.symmetries_of(descendent)
        for s, sym_state in enumerate(sym_states):

            # Empty macro if problem is solved in descendent state
            if domain.is_solved_in(sym_state): return actions, s, []

            # Non-empty macro if state matches a database pattern
            matched = pattern_database.query(sym_state)
            if matched: return actions, s, pattern_database.result()

    # Failure if no path to macro found
    return False

def run(state, domain, pattern_database, max_depth, max_macros):
    # returns plan = [...,(actions, sym index, macro),...] a sequence of macro_search results to the solved state
    # or plan = False if no plan is found
    
    # Form plan one macro at a time
    plan = []
    for num_macros in range(max_macros):

        # Search for next macro
        result = macro_search(state, domain, pattern_database, max_depth)
        
        # Return failure if none found
        if result is False: return False
        
        # Otherwise, execute search result
        actions, sym, macro = result
        for action in actions: state = domain.perform(action, state)
        state = domain.symmetries_of(state)[sym]
        for action in macro: state = domain.perform(action, state)
        plan.append(result)

        # Terminate once solved
        for sym_state in domain.symmetries_of(state):
            if domain.is_solved_in(state): return plan
    
    # At this point, no plan was found, so return failure
    return False

if __name__ == "__main__":

    max_depth = 2

    from cube import CubeDomain
    domain = CubeDomain(3)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    from pattern_database import PatternDatabase
    # patterns = domain.solved_state().reshape(1,-1)
    # macros = [((0,0,0),)]
    patterns = domain.perform((0,0,1), domain.solved_state()).reshape(1,-1)
    macros = [((0,0,3),)]
    pattern_database = PatternDatabase(patterns, macros)

    state = domain.solved_state()
    state = domain.perform((0,0,1),state)
    state = domain.symmetries_of(state)[2,:]
    state = domain.perform((1,0,1),state)
    result = macro_search(state, domain, bfs_tree, pattern_database, max_depth)
    if result != False:
        path, s, macro = result
        print(path)
        print(s)
        print(macro)

