
def path_thru_macro(state, problem, pattern_database, max_depth):

    if pattern_database.query(state): return pattern_database.result()
    if max_depth == 0: return False
    
    for action in problem.valid_actions(state):
        new_state = problem.perform(action, state)
        actions = path_thru_macro(new_state, problem, pattern_database, max_depth-1)
        if actions is not False: return [action] + actions
    
    return False

def attempt(state, problem, pattern_database, max_depth, max_macros):
    
    plan = []
    for num_macros in range(max_macros):

        actions = path_thru_macro(state, problem, pattern_database, max_depth)
        if actions is False: return False
        plan += actions
        
        for action in actions: state = problem.perform_(action, state)
        if problem.is_solved_in(state): return plan
    
    return False

if __name__ == "__main__":

    import cube as problem
