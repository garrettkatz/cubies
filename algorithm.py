
def path_to_macro(state, problem, pattern_database, max_depth):
    # returns result = (path, macro), each a sequence of actions
    # or result = False if there is no path to a macro

    # Empty path and macro if problem is already solved
    if problem.is_solved_in(state): return [], []

    # Empty path but non-macro if state matches a database pattern
    matched = pattern_database.query(state)
    if matched: return [], pattern_database.result()

    # Failure if max search depth is reached
    if max_depth == 0: return False    

    # Recursive search for remaining path to macro
    for action in problem.valid_actions(state):
        new_state = problem.perform(action, state)
        result = path_to_macro(new_state, problem, pattern_database, max_depth-1)
        if result is not False:
            path, macro = result
            return [action] + path, macro
    
    # Failure if recursive search did not find a path to a macro
    return False

def attempt(state, problem, pattern_database, max_depth, max_macros):
    # returns result = plan, a sequence of actions to the solved state
    # or result = False if no plan is found
    
    # Form plan one macro at a time
    plan = []
    for num_macros in range(max_macros):

        # Find path to next macro
        result = path_to_macro(state, problem, pattern_database, max_depth)
        
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

    import cube as problem

    state = problem.solved_state(3)
    problem.rotx_(state, depth=2, num_turns=1)
    pattern_database = problem.PatternDatabase()

    path, macro = path_to_macro(state, problem, pattern_database, max_depth=2)
    print(path)

