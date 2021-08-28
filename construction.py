import numpy as np
from pattern_database import PatternDatabase
from algorithm import run
import matplotlib.pyplot as pt

def pdb_query(state, patterns, wildcards):

    # # brute
    # # index = np.flatnonzero(((state == patterns) | wildcards).all(axis=1))
    # index = np.flatnonzero(((state[3:] == patterns[:,3:]) | wildcards[:,3:]).all(axis=1)) # first three facies invariant

    # progressive
    index = np.flatnonzero((state[3] == patterns[:,3]) | wildcards[:,3]) # first three facies invariant
    for k in range(4, patterns.shape[1]):
        if len(index) == 0: return index
        index = index[(state[k] == patterns[index, k]) | wildcards[index, k]]

    return index

class Constructor:

    def __init__(self, max_rules, max_incs, rng, domain, tree, max_depth, use_safe_depth, color_neutral):
        self.max_rules = max_rules
        self.max_incs = max_incs
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
        self.num_incs = 1

        # progress logging
        self.inc_added = np.ones(max_rules, dtype=int) * (max_incs + 1)
        self.inc_disabled = np.ones(self.wildcards.shape, dtype=int) * (max_incs + 1)
        self.inc_added[0] = 0
        self.inc_disabled[0] = 0

    def logs(self):
        return self.num_rules, self.num_incs, self.inc_added[:self.num_rules], self.inc_disabled[:self.num_rules]

    def rules(self):
        return self.patterns[:self.num_rules], self.wildcards[:self.num_rules], self.macros[:self.num_rules]

    def toggle_wildcard(self, triggered, state, path):
        # patterns, wildcards, macros = self.patterns, self.wildcards, self.macros
        patterns, wildcards, macros = self.rules()

        augmented = False
        for r in triggered: # query method
            goodmacro = (len(macros[r]) <= len(path)) and macros[r] == path[:len(macros[r])]
            if not goodmacro:
                w = self.rng.choice(np.flatnonzero(state != patterns[r]))
                wildcards[r, w] = False
                augmented = True
                self.inc_disabled[r, w] = self.num_incs

        return augmented

    def restrict_rules(self, state, path):

        augmented = False # becomes True if candidate gets augmented

        # patterns = self.patterns[:self.num_rules]
        # wildcards = self.wildcards[:self.num_rules]
        # macros = self.macros[:self.num_rules]
        patterns, wildcards, macros = self.rules()

        # restrict any rules needed so that state will not trigger bad macros
        # proved that neutral recolorings need not be considered in this step
        # (a non-reoriented optimal path to the recoloring exists and will enact the restriction)

        # triggered = np.flatnonzero(((state == patterns) | wildcards).all(axis=1))
        triggered = pdb_query(state, patterns, wildcards)

        toggled = self.toggle_wildcard(triggered, state, path)
        augmented |= toggled
        
        return augmented

    def has_neighboring_trigger(self, state, path):
        # check if state is in neighborhood of a trigger
        # due to incomplete tree it must also be triggered within distance to tree_depth
        # otherwise macro_search could exit set where pdb is correct

        # patterns = self.patterns[:self.num_rules]
        # wildcards = self.wildcards[:self.num_rules]
        # macros = self.macros[:self.num_rules]
        patterns, wildcards, macros = self.rules()

        safe_depth = max_depth
        if self.use_safe_depth: safe_depth = min(self.max_depth, self.tree.depth() - len(path))
        if self.color_neutral:
            for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
                for recoloring in self.domain.color_neutral_to(neighbor):
                    # triggered = ((recoloring == patterns) | wildcards).all(axis=1).any()
                    triggered = len(pdb_query(recoloring, patterns, wildcards)) > 0
                    if triggered: return True
        else:
            for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
                # triggered = ((neighbor == patterns) | wildcards).all(axis=1).any()
                triggered = len(pdb_query(neighbor, patterns, wildcards)) > 0
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
        self.inc_added[self.num_rules] = self.num_incs
        self.inc_disabled[self.num_rules, ~wildcard] = self.num_incs
        self.num_rules += 1

    def incorporate(self, state, path):
        augmented = self.restrict_rules(state, path)
        triggered = self.has_neighboring_trigger(state, path)
        if not triggered:
            augmented = True
            self.create_new_rule(state, path)
        self.num_incs += 1
        return augmented

def rewind(patterns, macros, inc_added, inc_disabled, inc):
    r = np.flatnonzero(inc_added <= inc).max() + 1
    patterns = patterns[:r]
    wildcards = (inc_disabled[:r] > inc)
    macros = macros[:r]
    return patterns, wildcards, macros

if __name__ == "__main__":

    cube_size, num_twist_axes, quarter_turns = 2, 2, True # 29k states

    # config
    # tree_depth = 11
    # use_safe_depth = True
    tree_depth = 15
    use_safe_depth = False
    max_depth = 1
    max_actions = 30
    color_neutral = True
    # breakpoint = 8000
    breakpoint = -1

    dump_period = 1000
    verbose = True

    do_cons = False
    show_results = False
    confirm = True

    import itertools as it
    from cube import CubeDomain
    valid_actions = tuple(it.product(range(num_twist_axes), range(1,cube_size), range(2-quarter_turns, 4, 2-quarter_turns)))
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    states = tree.states_rooted_at(init)
    paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved

    import pickle as pk

    if do_cons:

        max_rules = len(states)
        max_incs = max_rules * 10
        rng = np.random.default_rng()
        constructor = Constructor(max_rules, max_incs, rng, domain, tree, max_depth, use_safe_depth, color_neutral)
        inc_states = [0] # started with one rule at solved state

        done = False
        import itertools as it
        for epoch in it.count():
            if constructor.num_rules in [max_rules, breakpoint]: break
            if done: break
            done = True
    
            shuffler = np.random.permutation(range(len(states)))
            # shuffler = np.arange(len(states))): # solved outwards
            # shuffler = np.array(reversed(range(len(states)))): # outwards in
            for k,s in enumerate(shuffler):

                if constructor.num_rules in [max_rules, breakpoint]: break
                if verbose and k % (10**min(3, int(np.log10(k+1)))) == 0:
                    wildcards = constructor.rules()[1]
                    print("pass %d: %d <= %d rules (%d states), %f wildcard, done=%s (k=%d)" % (
                        epoch, constructor.num_rules, max_rules, len(states),
                        wildcards.sum() / wildcards.size, done, k))
    
                augmented = constructor.incorporate(states[s], paths[s])
                inc_states.append(s)
                if augmented: done = False
                
                if k % dump_period == 0:
                    with open("consres.pkl","wb") as f:
                        pk.dump((constructor.rules(), constructor.logs(), inc_states), f)
    
        if verbose: print("(max_depth = %d)" % max_depth)
    
        patterns, wildcards, macros = constructor.rules()
        with open("consres.pkl","wb") as f: pk.dump((constructor.rules(), constructor.logs(), inc_states), f)

        # np.set_printoptions(linewidth=200)
        # for k in range(10): print(patterns[k])
        # for k in range(10): print(patterns[-k])

    if show_results:

        with open("consres.pkl","rb") as f: (rules, logs, inc_states) = pk.load(f)
        patterns, wildcards, macros = rules
        num_rules, num_incs, inc_added, inc_disabled = logs

        bad_triggers = np.cumsum([(inc_disabled[inc_added < i] == i).any() for i in range(num_incs)])
        no_triggers = np.cumsum([(inc_added == i).any() for i in range(num_incs)])
        augmented = np.cumsum([(inc_disabled[inc_added < i] == i).any() or (inc_added == i).any() for i in range(num_incs)])

        # pt.subplot(1,3,1)
        # pt.plot(np.arange(num_incs), [(inc_added <= i).sum() for i in range(num_incs)])
        # pt.xlabel("iter")
        # pt.ylabel("num rules")
        # pt.subplot(1,3,2)
        # pt.plot(np.arange(num_incs), [(inc_disabled[inc_added <= i] > i).sum() for i in range(num_incs)])
        # pt.xlabel("iter")
        # pt.ylabel("num wildcards")
        # pt.subplot(1,3,3)
        # pt.plot(np.arange(num_incs), augmented)
        # pt.plot(np.arange(num_incs), bad_triggers)
        # pt.plot(np.arange(num_incs), no_triggers)
        # pt.legend(["aug", "bad trig", "new rule"])
        # pt.xlabel("iter")
        # pt.ylabel("num augmentations")
        # pt.show()

        num_probs = 16
        cats = ["sofar", "recent", "all"]
        correctness = {cat: list() for cat in cats}
        godliness = {cat: list() for cat in cats}
        folkliness = {cat: list() for cat in cats}
        converge_inc = np.argmax(np.cumsum(augmented))
        rewind_incs = np.linspace(num_probs, converge_inc, 30).astype(int)
        # rewind_incs = np.linspace(num_probs, num_incs, 30).astype(int)
        for rewind_inc in rewind_incs:

            rew_patterns, rew_wildcards, rew_macros = rewind(patterns, macros, inc_added, inc_disabled, rewind_inc)
            pdb = PatternDatabase(rew_patterns, rew_wildcards, rew_macros, domain)

            for cat in cats:

                num_solved = 0
                opt_moves = []
                alg_moves = []
            
                if cat == "sofar": probs = np.random.choice(rewind_inc, size=num_probs) # states so far, up to rewind_inc
                if cat == "recent": probs = np.arange(rewind_inc-num_probs, rewind_inc) # moving average near rewind_inc
                if cat == "all": probs = np.random.choice(len(states), size=num_probs) # all states

                for p in probs:
                    state, path = states[inc_states[p]], paths[inc_states[p]]
                    solved, plan = run(state, domain, tree, pdb, max_depth, max_actions, color_neutral)
                    num_solved += solved            
                    if solved and len(path) > 0:
                        opt_moves.append(len(path))
                        alg_moves.append(sum([len(a)+len(m) for a,_,m in plan]))
                
                correctness[cat].append( num_solved / num_probs )
                godliness[cat].append( np.mean( (np.array(opt_moves) + 1) / (np.array(alg_moves) + 1) ) )
                folkliness[cat].append( 1 -  len(rew_macros) / len(states) )

        for c, cat in enumerate(cats):
            pt.subplot(1,3, c+1)
            pt.plot(rewind_incs, correctness[cat], marker='o', label="correctness")
            pt.plot(rewind_incs, godliness[cat], marker='o', label="godliness")
            pt.plot(rewind_incs, folkliness[cat], marker='o', label="folkliness")
            pt.xlabel("num incs")
            pt.ylabel("performance")
            pt.ylim([0, 1.1])
            pt.legend()
            pt.title(cat)
        pt.show()

    # confirm correctness
    if confirm:
        with open("consres.pkl","rb") as f: (rules, logs, inc_states) = pk.load(f)
        patterns, wildcards, macros = rules
        num_rules, num_incs, inc_added, inc_disabled = logs

        # rewind = 100
        # patterns, wildcards, macros = rewind(patterns, macros, inc_added, inc_disabled, rewind)

        pdb = PatternDatabase(patterns, wildcards, macros, domain)    
        num_checked = 0
        num_solved = 0
        opt_moves = []
        alg_moves = []
        for p, (path, prob_state) in enumerate(tree.rooted_at(init)):
            num_checked += 1
            if verbose and p % (10**min(3, int(np.log10(p+1)))) == 0: print("checked %d of %d" % (num_checked, len(states)))
    
            # solved, plan = run(prob_state, domain, tree, pdb, max_depth, max_actions, color_neutral)
            solved, plan, _, _ = run(prob_state, domain, tree, pdb, max_depth, max_actions, color_neutral)
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

