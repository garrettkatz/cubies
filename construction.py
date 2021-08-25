import numpy as np
import matplotlib.pyplot as pt

class Constructor:

    def __init__(self, max_rules, rng, domain, tree, max_depth, use_safe_depth, color_neutral):
        self.max_rules = max_rules
        self.domain = domain
        self.rng = rng
        self.tree = tree
        self.max_depth = max_depth
        self.use_safe_depth = use_safe_depth
        self.color_neutral = color_neutral

        self.patterns = np.empty((max_rules, domain.state_size()), dtype=int)
        self.wildcards = np.empty(self.patterns.shape, dtype=bool)
        self.macros = [None] * max_rules

        self.patterns[0] = domain.solved_state()
        self.wildcards[0] = False
        self.macros[0] = ()

        self.num_rules = 1

    def rules(self):
        return self.patterns[:self.num_rules], self.wildcards[:self.num_rules], self.macros[:self.num_rules]

    def toggle_wildcard(self, triggered, state, path):
        patterns, wildcards, macros = self.patterns, self.wildcards, self.macros

        augmented = False
        for r in np.flatnonzero(triggered):
            goodmacro = (len(macros[r]) <= len(path)) and macros[r] == path[:len(macros[r])]
            if not goodmacro:
                wildcards[r, self.rng.choice(np.flatnonzero(state != patterns[r]))] = False
                augmented = True

        return augmented

    def restrict_rules(self, state, path):

        augmented = False # becomes True if candidate gets augmented

        patterns = self.patterns[:self.num_rules]
        wildcards = self.wildcards[:self.num_rules]
        macros = self.macros[:self.num_rules]

        # restrict any rules needed so that state will not trigger bad macros
        # proved that neutral recolorings need not be considered in this step
        # (a non-reoriented optimal path to the recoloring exists and will enact the restriction)
        triggered = ((state == patterns) | wildcards).all(axis=1)
        augmented |= self.toggle_wildcard(triggered, state, path)
        
        return augmented

    def has_neighboring_trigger(self, state, path):
        # check if state is in neighborhood of a trigger
        # due to incomplete tree it must also be triggered within distance to tree_depth
        # otherwise macro_search could exit set where pdb is correct

        patterns = self.patterns[:self.num_rules]
        wildcards = self.wildcards[:self.num_rules]
        macros = self.macros[:self.num_rules]

        safe_depth = max_depth
        if self.use_safe_depth: safe_depth = min(self.max_depth, self.tree.depth() - len(path))
        if self.color_neutral:
            for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
                for recoloring in self.domain.color_neutral_to(neighbor):
                    triggered = ((recoloring == patterns) | wildcards).all(axis=1).any()
                    if triggered: return True
        else:
            for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
                triggered = ((neighbor == patterns) | wildcards).all(axis=1).any()
                if triggered: return True
            # # slower in early profile despite vectorization
            # neighbors = self.tree.states_rooted_at(state, up_to_depth=safe_depth)
            # neighbors = neighbors[:, np.newaxis, :]
            # triggered = ((neighbors == patterns) | wildcards).all(axis=2).any()
            # if triggered: return True
        return False

    def create_new_rule(self, state, path):
        # if this code is reached, path is longer than max depth
        macro = path[:self.rng.integers(self.max_depth, len(path))+1] # random macro    
        # macro = path
        pattern = state
        # wildcard = np.ones(pattern.shape, dtype=bool) # start with all wildcards which will gradually be disabled
        wildcard = (np.random.rand(*pattern.shape) < (len(path) / self.domain.god_number())) # more wildcards in deeper states

        # add to pdb
        self.patterns[self.num_rules] = pattern
        self.wildcards[self.num_rules] = wildcard
        self.macros[self.num_rules] = macro
        self.num_rules += 1

    def incorporate(self, state, path):
        augmented = self.restrict_rules(state, path)
        triggered = self.has_neighboring_trigger(state, path)
        if not triggered:
            augmented = True
            self.create_new_rule(state, path)
        return augmented
    
if __name__ == "__main__":

    # config
    tree_depth = 11
    use_safe_depth = False
    max_depth = 1
    cube_size = 2
    max_actions = 30
    color_neutral = True
    # breakpoint = 8000
    breakpoint = -1

    dump_period = 1000
    verbose = True

    do_cons = True
    show_results = False
    confirm = True

    from cube import CubeDomain
    domain = CubeDomain(cube_size)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    states = tree.states_rooted_at(init)
    paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved

    import pickle as pk

    if do_cons:

        max_rules = len(states)
        rng = np.random.default_rng()
        constructor = Constructor(max_rules, rng, domain, tree, max_depth, use_safe_depth, color_neutral)
    
        # # initialize one rule for solved state
        # patterns = states[:1,:]
        # wildcards = np.zeros(patterns.shape, dtype=bool)
        # macros = [()]
    
        rule_counts = []
        wildcard_counts = []
    
        done = False
        import itertools as it
        for epoch in it.count():
            if constructor.num_rules in [max_rules, breakpoint]: break
            if done: break
            done = True
    
            for k,s in enumerate(np.random.permutation(range(len(states)))):
            # for k,s in enumerate(range(len(states))): # solved outwards
            # for k,s in enumerate(reversed(range(len(states)))): # outwards in

                patterns, wildcards, macros = constructor.rules()
                if len(macros) in [max_rules, breakpoint]: break
                if verbose and k % (10**min(3, int(np.log10(k+1)))) == 0:
                    print("pass %d: %d <= %d rules (%d states), %f wildcard, done=%s (k=%d)" % (
                        epoch, len(macros), max_rules, len(states), wildcards.sum() / wildcards.size, done, k))
                rule_counts.append(len(macros))
                wildcard_counts.append(wildcards.sum())
    
                augmented = constructor.incorporate(states[s], paths[s])
                if augmented: done = False
                
                if len(rule_counts) % dump_period == 0:
                    with open("consres.pkl","wb") as f: pk.dump((patterns, wildcards, macros, rule_counts, wildcard_counts), f)
    
        if verbose: print("(max_depth = %d)" % max_depth)
    
        patterns, wildcards, macros = constructor.rules()
        with open("consres.pkl","wb") as f: pk.dump((patterns, wildcards, macros, rule_counts, wildcard_counts), f)

    if show_results:
        with open("consres.pkl","rb") as f: (patterns, wildcards, macros, rule_counts, wildcard_counts) = pk.load(f)
        import matplotlib.pyplot as pt
        pt.subplot(1,2,1)
        pt.plot((np.array(rule_counts) + 1))
        pt.xlabel("iter")
        pt.ylabel("num rules")
        pt.subplot(1,2,2)
        pt.plot((np.array(wildcard_counts) + 1))
        pt.xlabel("iter")
        pt.ylabel("num wildcards")
        pt.show()

    # confirm correctness
    if confirm:
        with open("consres.pkl","rb") as f: (patterns, wildcards, macros, rule_counts, wildcard_counts) = pk.load(f)

        from pattern_database import PatternDatabase
        pdb = PatternDatabase(patterns, wildcards, macros, domain)
    
        from algorithm import run
        import matplotlib.pyplot as pt
    
        num_checked = 0
        num_solved = 0
        opt_moves = []
        alg_moves = []
        for p, (path, prob_state) in enumerate(tree.rooted_at(init)):
            num_checked += 1
            if verbose and p % (10**min(3, int(np.log10(p+1)))) == 0: print("checked %d of %d" % (num_checked, len(states)))
    
            solved, plan = run(prob_state, domain, tree, pdb, max_depth, max_actions, color_neutral)
            num_solved += solved

            state = prob_state
            for (actions, sym, macro) in plan:
                state = domain.execute(actions, state)
                if color_neutral: state = domain.color_neutral_to(state)[sym]
                state = domain.execute(macro, state)
    
            if len(path) > 0:
                opt_moves.append(len(path))
                alg_moves.append(sum([len(a)+len(m) for a,_,m in plan]))
    
            if not domain.is_solved_in(state):

                print(len(macros))

                numcols = 20
                numrows = 6
                state = prob_state
                ax = domain.render_subplot(numrows,numcols,1,state)
                ax.set_title("opt path")
                for a, action in enumerate(domain.reverse(path)):
                    state = domain.perform(action, state)
                    domain.render_subplot(numrows,numcols,a+2,state)
    
                state = prob_state
                ax = domain.render_subplot(numrows,numcols,numcols+1,state)
                ax.set_title("alg path")
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
                    if color_neutral:
                        state = domain.color_neutral_to(state)[sym]
                        ax = domain.render_subplot(numrows,numcols, sp, state)
                        ax.set_title(str(sym))
                        sp += 1
                    for a,action in enumerate(macro):
                        state = domain.perform(action, state)
                        ax = domain.render_subplot(numrows,numcols, sp, state)
                        ax.set_title(str(action) + "|")
                        sp += 1
    
                pt.show()
    
            assert solved
    
        alg_moves = np.array(alg_moves[1:]) # skip solved state from metrics
        opt_moves = np.array(opt_moves[1:])
        alg_opt = alg_moves / opt_moves
        if verbose: print("alg/opt min,max,mean", (alg_opt.min(), alg_opt.max(), alg_opt.mean()))
        if verbose: print("alg min,max,mean", (alg_moves.min(), alg_moves.max(), alg_moves.mean()))
        if verbose: print("Solved %d (%d/%d = %f)" % (num_solved, len(patterns), num_checked, len(patterns)/num_checked))

