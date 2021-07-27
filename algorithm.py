def macro_search(state, problem, bfs_tree, pattern_database, max_depth):
    # returns result = (actions, macro)
    # or result = False if there is no path to a macro
    
    for actions, permutation in bfs_tree:

        # Don't exceed maximum search depth
        if len(actions) > max_depth: continue    

        # Compute descendent state
        descendent = state[permutation]

        # Empty macro if problem is solved in descendent state
        if problem.is_solved_in(descendent): return actions, []
        
        # Non-empty macro if state matches a database pattern
        matched = pattern_database.query(descendent)
        if matched: return actions, pattern_database.result()

    # Failure if no path to macro found
    return False

def attempt(state, problem, pattern_database, max_depth, max_macros):
    # returns result = plan, a sequence of actions to the solved state
    # or result = False if no plan is found
    
    # Form plan one macro at a time
    plan = []
    for num_macros in range(max_macros):

        # Find path to next macro
        result = macro_search(state, problem, pattern_database, max_depth)
        
        # Return failure if none found
        if result is False: return False
        
        # Otherwise, execute path and macro, one action at a time
        path, macro = result
        actions = path + macro
        for action in actions:

            # Update the plan with the action
            plan.append(action)
            
            # Update the state after performing the action
            state = problem.perform_(action, state)
            
            # Terminate as soon as the problem is solved
            if problem.is_solved_in(state): return plan
    
    # At this point, no plan was found, so return failure
    return False

if __name__ == "__main__":

    from search_tree import SearchTree

    import cube as problem

    state = problem.solved_state(3)
    problem.rotx_(state, depth=2, num_turns=1)
    pattern_database = problem.PatternDatabase()

    path, macro = macro_search(state, problem, pattern_database, max_depth=2)
    print(path)

