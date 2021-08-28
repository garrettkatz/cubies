import numpy as np
from pattern_database import PatternDatabase, pdb_query
from algorithm import run
import matplotlib.pyplot as pt

class Constructor:

    def __init__(self, max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral):
        self.max_rules = max_rules
        self.domain = domain
        self.rng = rng
        self.tree = tree
        self.max_depth = max_depth
        self.max_actions = max_actions
        self.use_safe_depth = use_safe_depth
        self.color_neutral = color_neutral

        self.patterns = np.empty((max_rules, domain.state_size()), dtype=int)
        self.wildcards = np.empty(self.patterns.shape, dtype=bool)
        self.macros = [None] * max_rules
        self.chain_lengths = np.empty(max_rules, dtype=int)

        self.patterns[0] = domain.solved_state()
        self.wildcards[0] = False
        self.macros[0] = ()
        self.chain_lengths[0] = 0

        self.num_rules = 1
        self.num_incs = 1

        # progress logging
        self.inc_added = np.ones(max_rules, dtype=int) * np.iinfo(int).max
        self.inc_disabled = np.ones(self.wildcards.shape, dtype=int) * np.iinfo(int).max
        self.inc_added[0] = 0
        self.inc_disabled[0] = 0

    def logs(self):
        return self.num_rules, self.num_incs, self.inc_added[:self.num_rules], self.inc_disabled[:self.num_rules], self.chain_lengths[:self.num_rules]

    def rules(self):
        return self.patterns[:self.num_rules], self.wildcards[:self.num_rules], self.macros[:self.num_rules]

    def toggle_wildcard(self, triggered, state, path):
        patterns, wildcards, macros = self.rules()
        pdb = PatternDatabase(patterns, wildcards, macros, self.domain)

        toggled = False
        for r in triggered:

            # check if trigger is first link of a failed algorithm macro chain
            new_state = self.domain.execute(macros[r], state)
            max_actions = self.max_actions - len(macros[r]) - self.max_depth # subtract steps for first macro and its neighborhood
            solved, plan, rules, triggerers = run(new_state, self.domain, self.tree, pdb, self.max_depth, max_actions, self.color_neutral)

            if not solved:

                rules.insert(0, r)
                triggerers.insert(0, state)

                # new rule creation ensures that at least one chain link has wildcards that can be toggled

                # t, rule = 0, r # restrict first trigger
                links_with_wildcards = [t for t in range(len(rules)) if (triggerers[t] != patterns[rules[t]]).any()]
                t = self.rng.choice(links_with_wildcards)
                rule = rules[t]

                w = self.rng.choice(np.flatnonzero(triggerers[t] != patterns[rule]))
                wildcards[rule, w] = False
                toggled = True
                self.inc_disabled[rule, w] = self.num_incs

        return toggled

    # def toggle_wildcard(self, triggered, state, path):
    #     # patterns, wildcards, macros = self.patterns, self.wildcards, self.macros
    #     patterns, wildcards, macros = self.rules()

    #     toggled = False
    #     for r in triggered: # query method
    #         goodmacro = (len(macros[r]) <= len(path)) and macros[r] == path[:len(macros[r])]
    #         if not goodmacro:
    #             w = self.rng.choice(np.flatnonzero(state != patterns[r]))
    #             wildcards[r, w] = False
    #             toggled = True
    #             self.inc_disabled[r, w] = self.num_incs

    #     return toggled

    def restrict_rules(self, state, path):
        patterns, wildcards, macros = self.rules()
        triggered = pdb_query(state, patterns, wildcards)
        toggled = self.toggle_wildcard(triggered, state, path)
        return toggled

    def has_neighboring_trigger(self, state, path):
        # check if any recoloring of state neighbors a trigger

        patterns, wildcards, macros = self.rules()

        # for incomplete trees it must also be triggered within distance to tree_depth
        # otherwise macro_search could exit set where pdb is correct
        safe_depth = self.max_depth
        if self.use_safe_depth: safe_depth = min(self.max_depth, self.tree.depth() - len(path))

        # search all descendents for one recoloring before starting the next
        # first recoloring is the identity
        if self.color_neutral:
            recolorings = self.domain.color_neutral_to(state)
        else:
            recolorings = state.reshape(1, self.domain.state_size())

        # search all neighbor for triggers
        for recoloring in recolorings:
            for neighbor in self.tree.states_rooted_at(recoloring, up_to_depth=safe_depth):
                triggered = len(pdb_query(neighbor, patterns, wildcards)) > 0
                if triggered: return True
        return False

    # def has_neighboring_trigger(self, state, path):
    #     # check if state is in neighborhood of a trigger
    #     # due to incomplete tree it must also be triggered within distance to tree_depth
    #     # otherwise macro_search could exit set where pdb is correct

    #     patterns, wildcards, macros = self.rules()
    #     safe_depth = self.max_depth
    #     if self.use_safe_depth: safe_depth = min(self.max_depth, self.tree.depth() - len(path))
    #     if self.color_neutral:
    #         for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
    #             for recoloring in self.domain.color_neutral_to(neighbor):
    #                 triggered = len(pdb_query(recoloring, patterns, wildcards)) > 0
    #                 if triggered: return True
    #     else:
    #         for neighbor in self.tree.states_rooted_at(state, up_to_depth=safe_depth):
    #             triggered = len(pdb_query(neighbor, patterns, wildcards)) > 0
    #             if triggered: return True
    #     return False

    def create_new_rule(self, state, path):
        patterns, wildcards, macros = self.rules()

        # identify interstates that match full patterns in the pdb
        interstates = self.domain.intermediate_states(path, state)
        new_macro_lengths = []
        rule_indices = []
        for i, interstate in enumerate(interstates):            

            # find the full matching pattern, if any
            r = np.flatnonzero((interstate == patterns).all(axis=1))
            if len(r) != 1: continue
            r = r[0]

            # mark interstates that could serve as next link in a successful algorithm macro chain
            new_macro_length = i+1
            if self.chain_lengths[r] + new_macro_length <= self.max_actions:
                new_macro_lengths.append(new_macro_length)
                rule_indices.append(r)

        # add a new macro that goes from current pattern to another full pattern in pdb
        # ensures that there is always at least one wildcard to toggle when restricting rules
        k = self.rng.choice(len(new_macro_lengths))
        new_macro = path[:new_macro_lengths[k]]
        new_chain_length = new_macro_lengths[k] + self.chain_lengths[rule_indices[k]]

        pattern = state

        # wildcard = np.ones(pattern.shape, dtype=bool) # start with all wildcards which will gradually be disabled
        # wildcard = np.zeros(pattern.shape, dtype=bool) # pdb with no wildcards
        wildcard = self.rng.uniform(size=pattern.shape) > (1 / len(path)) # more wildcards in deeper states
        # wildcard = (np.random.rand(*pattern.shape) < (len(path) / self.domain.god_number())) # but relies on accurate path length and god_number

        # add to pdb
        self.patterns[self.num_rules] = pattern
        self.wildcards[self.num_rules] = wildcard
        self.macros[self.num_rules] = new_macro
        self.chain_lengths[self.num_rules] = new_chain_length

        self.inc_added[self.num_rules] = self.num_incs
        self.inc_disabled[self.num_rules, ~wildcard] = self.num_incs
        self.num_rules += 1

    # def create_new_rule(self, state, path):
    #     # if this code is reached, path is longer than max depth
    #     macro = path[:self.rng.integers(self.max_depth, len(path))+1] # random macro
    #     # assert len(macro) > 0
    #     pattern = state
    #     # wildcard = np.ones(pattern.shape, dtype=bool) # start with all wildcards which will gradually be disabled

    #     # macro = path
    #     # wildcard = np.zeros(pattern.shape, dtype=bool)
    #     wildcard = (np.random.rand(*pattern.shape) < (len(path) / self.domain.god_number())) # more wildcards in deeper states, but god_number wrong for easier cube variants

    #     # add to pdb
    #     self.patterns[self.num_rules] = pattern
    #     self.wildcards[self.num_rules] = wildcard
    #     self.macros[self.num_rules] = macro
    #     self.inc_added[self.num_rules] = self.num_incs
    #     self.inc_disabled[self.num_rules, ~wildcard] = self.num_incs
    #     self.num_rules += 1

    def incorporate(self, state, path):
        toggled = self.restrict_rules(state, path)
        has_trigger = self.has_neighboring_trigger(state, path)
        if not has_trigger: self.create_new_rule(state, path)
        self.num_incs += 1
        return toggled or not has_trigger

    def evaluate(self, probs):
        patterns, wildcards, macros = self.rules()
        pdb = PatternDatabase(patterns, wildcards, macros, self.domain)

        num_solved = 0
        godlinesses =  []    
        for p, (state, path) in enumerate(probs):
            solved, plan, _, _ = run(state, self.domain, self.tree, pdb, self.max_depth, self.max_actions, self.color_neutral)
            num_solved += solved
            opt_bound = min(self.domain.god_number(), len(path))
            alg_moves = sum([len(a)+len(m) for _,a,m in plan])
            godlinesses.append(0 if not solved else (opt_bound+1) / (alg_moves+1))
        
        correctness = num_solved / len(probs)
        godliness = np.mean(godlinesses)
        folkliness = 1 - self.num_rules / self.max_rules

        return correctness, godliness, folkliness

def rewind(patterns, macros, inc_added, inc_disabled, inc):
    r = np.flatnonzero(inc_added <= inc).max() + 1
    patterns = patterns[:r]
    wildcards = (inc_disabled[:r] > inc)
    macros = macros[:r]
    return patterns, wildcards, macros

if __name__ == "__main__":

    # config
    cube_size, num_twist_axes, quarter_turns = 2, 2, True # 29k states
    # cube_size, num_twist_axes, quarter_turns = 2, 3, False # 24 states

    tree_depth = 14
    use_safe_depth = False
    max_depth = 1
    max_actions = 30
    color_neutral = True

    num_problems = 32

    breakpoint = -1
    # breakpoint = 100
    num_reps = 1
    verbose = True

    do_cons = True
    show_results = False
    confirm = True

    # set up descriptive dump name
    dump_period = 1000
    dump_dir = "rcons"
    dump_base = "N%da%dq%d_D%d_M%d_cn%d" % (cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral)

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

    max_rules = len(states)

    import pickle as pk
    import os
    import itertools as it

    if do_cons:

        for rep in range(num_reps):

            rng = np.random.default_rng()
            constructor = Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
            inc_states = [0] # started with one rule at solved state
    
            done = False
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
                    # if verbose:
    
                        probs = [(states[p], paths[p]) for p in np.random.choice(len(states), size = num_problems)]
                        correctness, godliness, _ = constructor.evaluate(probs)
    
                        wildcards = constructor.rules()[1]
                        print("%d,%d,%d: %d <= %d rules (%d states), %f solved, %f godly, %f wildcard, done=%s" % (
                            rep, epoch, k, constructor.num_rules, max_rules, len(states),
                            correctness, godliness,
                            wildcards.sum() / wildcards.size, done))
        
                    augmented = constructor.incorporate(states[s], paths[s])
                    inc_states.append(s)
                    if augmented: done = False
                    
                    if k % dump_period == 0:
                        dump_name = "%s_r%d" % (dump_base, rep)
                        with open(dump_name + ".pkl", "wb") as f:
                            pk.dump((constructor.rules(), constructor.logs(), inc_states), f)
        
            if verbose: print("(max_depth = %d)" % max_depth)
    
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((constructor.rules(), constructor.logs(), inc_states), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))
    
            # patterns, wildcards, macros = constructor.rules()
            # np.set_printoptions(linewidth=200)
            # for k in range(10): print(patterns[k])
            # for k in range(10): print(patterns[-k])

    if show_results:

        rep = 0
        dump_name = "%s_r%d" % (dump_base, rep)
        print(dump_name)
        with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, inc_states) = pk.load(f)
        patterns, wildcards, macros = rules
        num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        # fix for max_incs vestigial
        print("remove this!")
        inc_disabled[inc_disabled == 10*len(states) + 1] = np.iinfo(int).max

        ### show pdb
        numrows = min(14, len(macros))
        numcols = min(15, max(map(len, macros)) + 2)
        for r in range(numrows):
            rule = r if r < numrows/2 else len(patterns)-(numrows-r)
            ax = domain.render_subplot(numrows, numcols, r*numcols + 1, patterns[rule])
            if r == 0: ax.set_title("pattern")
            ax = domain.render_subplot(numrows, numcols, r*numcols + 2, patterns[rule] * (1 - wildcards[rule]))
            if r == 0: ax.set_title("trigger")
            else: ax.set_title(str(wildcards[rule].sum()))
            state = patterns[rule]
            for m in range(len(macros[rule])):
                if 2+m+1 > numcols: break
                state = domain.perform(macros[rule][m], state)
                ax = domain.render_subplot(numrows, numcols, r*numcols + 2 + m+1, state)
                ax.set_title(str(macros[rule][m]))
        pt.tight_layout()
        pt.show()

        wildcards_disabled = np.zeros(num_incs, dtype=int)
        rules_added = np.zeros(num_incs, dtype=int)
        for r in range(len(patterns)):
            for w in range(domain.state_size()):
                if inc_disabled[r,w] >= len(wildcards_disabled): continue
                wildcards_disabled[inc_disabled[r,w]] += 1
            rules_added[inc_added[r]] += 1
        augmented = (rules_added > 0) | (wildcards_disabled > 0)

        pt.subplot(1,4,1)
        # pt.plot(np.arange(num_incs), [(inc_added <= i).sum() for i in range(num_incs)])
        pt.plot(np.arange(num_incs), np.cumsum(rules_added))
        pt.xlabel("iter")
        pt.ylabel("num rules")
        pt.subplot(1,4,2)
        # pt.plot(np.arange(num_incs), [(inc_disabled[inc_added <= i] > i).sum() for i in range(num_incs)])
        pt.plot(np.arange(num_incs), np.cumsum(rules_added)*domain.state_size() - np.cumsum(wildcards_disabled))
        pt.xlabel("iter")
        pt.ylabel("num wildcards")
        pt.subplot(1,4,3)
        pt.plot(np.arange(num_incs), np.cumsum(augmented))
        pt.plot(np.arange(num_incs), np.cumsum(wildcards_disabled > 0))
        pt.plot(np.arange(num_incs), np.cumsum(rules_added > 0))
        pt.legend(["aug", "bad trig", "new rule"])
        pt.xlabel("iter")
        pt.ylabel("num augmentations")
        pt.subplot(1,4,4)
        # pt.plot(np.arange(num_rules), chain_lengths, 'k.')
        pt.hist(chain_lengths)
        pt.xlabel("rule")
        pt.ylabel("chain length")
        pt.show()

        num_problems = 16
        cats = ["sofar", "recent", "all"]
        correctness = {cat: list() for cat in cats}
        godliness = {cat: list() for cat in cats}
        folkliness = {cat: list() for cat in cats}
        converge_inc = np.argmax(np.cumsum(augmented))
        rewind_incs = np.linspace(num_problems, converge_inc, 30).astype(int)
        # rewind_incs = np.linspace(num_problems, num_incs, 30).astype(int)
        for rewind_inc in rewind_incs:

            rew_patterns, rew_wildcards, rew_macros = rewind(patterns, macros, inc_added, inc_disabled, rewind_inc)
            pdb = PatternDatabase(rew_patterns, rew_wildcards, rew_macros, domain)

            for cat in cats:

                if cat == "sofar": probs = np.random.choice(rewind_inc, size=num_problems) # states so far, up to rewind_inc
                if cat == "recent": probs = np.arange(rewind_inc-num_problems, rewind_inc) # moving average near rewind_inc
                if cat == "all": probs = np.random.choice(len(states), size=num_problems) # all states

                num_solved = 0
                opt_moves = []
                alg_moves = []
            
                for p in probs:
                    state, path = states[inc_states[p]], paths[inc_states[p]]
                    solved, plan, _, _ = run(state, domain, tree, pdb, max_depth, max_actions, color_neutral)
                    num_solved += solved            
                    if solved and len(path) > 0:
                        opt_moves.append(len(path))
                        alg_moves.append(sum([len(a)+len(m) for _,a,m in plan]))
                
                correctness[cat].append( num_solved / num_problems )
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
        rep = np.random.choice(num_reps)
        dump_name = "%s_r%d" % (dump_base, rep)
        with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, inc_states) = pk.load(f)
        patterns, wildcards, macros = rules
        num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        # rewind = 100
        # patterns, wildcards, macros = rewind(patterns, macros, inc_added, inc_disabled, rewind)

        pdb = PatternDatabase(patterns, wildcards, macros, domain)    
        num_checked = 0
        num_solved = 0
        opt_moves = []
        alg_moves = []
        # probs = [
        #     (((1,1,2),), domain.perform((1,1,2), domain.solved_state())),
        #     (((2,1,2),), domain.perform((2,1,2), domain.solved_state())),
        # ]
        probs = tree.rooted_at(init)
        for p, (path, prob_state) in enumerate(probs):
            num_checked += 1
            if verbose and p % (10**min(3, int(np.log10(p+1)))) == 0: print("checked %d of %d" % (num_checked, len(states)))
    
            solved, plan, rule_indices, interstates = run(prob_state, domain, tree, pdb, max_depth, max_actions, color_neutral)
            num_solved += solved

            state = prob_state
            for (sym, actions, macro) in plan:
                if color_neutral: state = domain.color_neutral_to(state)[sym]
                state = domain.execute(actions, state)
                state = domain.execute(macro, state)
            final_state = state
    
            if len(path) > 0:
                opt_moves.append(len(path))
                alg_moves.append(sum([len(a)+len(m) for _,a,m in plan]))
    
            if not solved:

                print("num actions:", sum([len(a)+len(m) for _,a,m in plan]))

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
                for p, (sym, actions, macro) in enumerate(plan):
                    print("actions, sym, macro, rule index")
                    print(actions, sym, macro,rule_indices[p])

                    if color_neutral:
                        state = domain.color_neutral_to(state)[sym]
                        ax = domain.render_subplot(numrows,numcols, sp, state)
                        ax.set_title(str(sym))
                        sp += 1

                    for a,action in enumerate(actions):
                        state = domain.perform(action, state)
                        ax = domain.render_subplot(numrows,numcols, sp, state)
                        if a == 0:
                            ax.set_title("|" + str(action))
                        else:
                            ax.set_title(str(action))
                        sp += 1

                    ax = domain.render_subplot(numrows,numcols, sp, patterns[rule_indices[p]] * (1 - wildcards[rule_indices[p]]))
                    ax.set_title("trig " + str(chain_lengths[rule_indices[p]]))
                    sp += 1

                    ax = domain.render_subplot(numrows,numcols, sp, patterns[rule_indices[p]])
                    ax.set_title("pattern")
                    sp += 1

                    for a,action in enumerate(macro):
                        state = domain.perform(action, state)
                        ax = domain.render_subplot(numrows,numcols, sp, state)
                        if a == len(macro)-1: ax.set_title(str(action) + "|")
                        else: ax.set_title(str(action))
                        sp += 1
    
                pt.show()
    
            assert solved == domain.is_solved_in(final_state)
            assert solved
    
        alg_moves = np.array(alg_moves[1:]) # skip solved state from metrics
        opt_moves = np.array(opt_moves[1:])
        alg_opt = alg_moves / opt_moves
        if verbose: print("alg/opt min,max,mean", (alg_opt.min(), alg_opt.max(), alg_opt.mean()))
        if verbose: print("alg min,max,mean", (alg_moves.min(), alg_moves.max(), alg_moves.mean()))
        if verbose: print("Solved %d (%d/%d = %f)" % (num_solved, len(patterns), num_checked, len(patterns)/num_checked))

