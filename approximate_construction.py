import numpy as np
from relaxed_construction import Constructor, rewind
from utils import softmax
from pattern_database import PatternDatabase
from algorithm import run
import matplotlib.pyplot as pt

def scrambled(domain, rng, max_scramble_length):
    # make scrambled path and state
    # make sure all states reachable (fixed scramble length might violate this)
    # make sure all paths are <= max_actions - max_depth so that new rule creation is always possible
    probs = np.arange(max_scramble_length+1, dtype=float)
    probs /= probs.sum()
    def scrambled_sample():
        scramble_length = rng.choice(max_scramble_length + 1, p=probs)
        scramble = list(map(tuple, rng.choice(domain.valid_actions(), size=scramble_length)))
        state = domain.execute(scramble, domain.solved_state())
        path = domain.reverse(scramble)
        return state, path
    return scrambled_sample

def uniform(rng, states, paths):
    def uniform_sample():
        k = rng.choice(len(states))
        return states[k], paths[k]
    return uniform_sample

if __name__ == "__main__":

    # config
    cube_size, num_twist_axes, quarter_turns = 3, 3, True # full cube
    # cube_size, num_twist_axes, quarter_turns = 2, 2, True # 29k states
    # cube_size, num_twist_axes, quarter_turns = 2, 3, False # 24 states

    tree_depth = 1
    use_safe_depth = False
    max_depth = 1
    max_actions = 30
    color_neutral = True

    # static_incs_for_stop = 256
    num_problems = 1024
    eval_period = 1000
    correctness_bar = 0.99
    inc_sampler = "scrambled"
    # eval_samplers = ["scrambled", "uniform"]
    eval_samplers = ["scrambled"]
    assert inc_sampler in eval_samplers

    breakpoint = -1
    # breakpoint = 100
    num_reps = 16
    break_seconds = 10 * 60
    verbose = True

    do_cons = True
    show_results = False
    confirm = False
    confirm_show = False

    # set up descriptive dump name
    dump_period = 1000
    dump_dir = "acons"
    dump_base = "N%da%dq%d_D%d_M%d_cn%d_%s_cb%s" % (
        cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral, inc_sampler, correctness_bar)
    # dump_base = "N%da%dq%d_D%d_M%d_cn%d" % (
    #     cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral)
    # dump_base = "N%da%dq%d_D%d_M%d_cn%d_T%d" % (
    #     cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral, static_incs_for_stop)

    import itertools as it
    from cube import CubeDomain
    valid_actions = tuple(it.product(range(num_twist_axes), range(1,cube_size), range(2-quarter_turns, 4, 2-quarter_turns)))
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    # all_states = tree.states_rooted_at(init)
    # optimal_paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved
    # max_rules = len(all_states)

    all_states = []
    optimal_paths = []
    max_rules = 100_000

    max_scramble_length = max_actions - max_depth

    import pickle as pk
    import os

    if do_cons:

        from time import sleep

        for rep in range(num_reps):

            rng = np.random.default_rng()
            constructor = Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
            inc_states = {domain.solved_state().tobytes(): [0]}

            uniform_sample = uniform(rng, all_states, optimal_paths)
            scrambled_sample = scrambled(domain, rng, max_scramble_length)
            
            scrambled_objectives = []
            uniform_objectives = []

            static_incs = 0
            max_static_incs = 0
            for k in it.count():
                if constructor.num_rules in [max_rules, breakpoint]: break
                # if static_incs == static_incs_for_stop: break

                # max_scramble_length = max_actions - max_depth
                # # probs = softmax(np.arange(max_scramble_length+1))
                # probs = np.arange(max_scramble_length+1, dtype=float)
                # probs /= probs.sum()
                # scramble_length = rng.choice(max_scramble_length + 1, p=probs)
                # scramble = list(map(tuple, rng.choice(domain.valid_actions(), size=scramble_length)))
                # state = domain.execute(scramble, domain.solved_state())
                # path = domain.reverse(scramble)

                if inc_sampler == "scrambled": state, path = scrambled_sample()
                if inc_sampler == "uniform": state, path = uniform_sample()

                state_bytes = state.tobytes()
                if state_bytes not in inc_states: inc_states[state_bytes] = []
                inc_states[state_bytes].append(k)

                augmented = constructor.incorporate(state, path)
                if augmented:
                    static_incs = 0
                else:
                    static_incs += 1
                    max_static_incs = max(max_static_incs, static_incs)

                if k % eval_period == 0:

                    if "scrambled" in eval_samplers:
                        probs = [scrambled_sample() for _ in range(num_problems)]
                        scrambled_objectives.append(constructor.evaluate(probs))

                    if "uniform" in eval_samplers:
                        probs = [uniform_sample() for _ in range(num_problems)]
                        uniform_objectives.append(constructor.evaluate(probs))

                    if inc_sampler == "scrambled": correctness, godliness, _ = scrambled_objectives[-1]
                    if inc_sampler == "uniform": correctness, godliness, _ = uniform_objectives[-1]

                    if verbose:
                        wildcards = constructor.rules()[1]
                        print("%d,%d: %d <= %d rules (%d states), %f solved, %f godly, %f wildcard, static for %d" % (
                            rep, k, constructor.num_rules, max_rules, len(all_states),
                            correctness, godliness,
                            wildcards.sum() / wildcards.size, static_incs))

                    if correctness >= correctness_bar: break

                if k % dump_period == 0:
                    dump_name = "%s_r%d" % (dump_base, rep)
                    with open(dump_name + ".pkl", "wb") as f:
                        pk.dump((constructor.rules(), constructor.logs(), (scrambled_objectives, uniform_objectives), inc_states), f)
        
            if verbose: print("(max_depth = %d)" % max_depth)
    
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((constructor.rules(), constructor.logs(), (scrambled_objectives, uniform_objectives), inc_states), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))
    
            # patterns, wildcards, macros = constructor.rules()
            # np.set_printoptions(linewidth=200)
            # for k in range(10): print(patterns[k])
            # for k in range(10): print(patterns[-k])

            if verbose: print("Breaking for %s seconds..." % str(break_seconds))
            sleep(break_seconds)

    if show_results:

        for rep in range(num_reps):
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, inc_states) = pk.load(f)
            num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

            incs = np.arange(0, num_incs, eval_period)
            for n, (name, objective) in enumerate(zip(["scrambled","uniform"], objectives)):
                correctness, godliness, folkliness = zip(*objective)
                pt.subplot(2,3,n*3 + 1)
                pt.plot(incs, correctness)
                pt.ylabel("correctness")
                pt.title(name)
                pt.xlabel("incs")
                pt.subplot(2,3,n*3 + 2)
                pt.plot(incs, godliness)
                pt.ylabel("godliness")
                pt.title(name)
                pt.xlabel("incs")
                pt.subplot(2,3,n*3 + 3)
                pt.plot(incs, folkliness)
                pt.ylabel("folkliness")
                pt.title(name)
                pt.xlabel("incs")
        pt.show()

        # rep = 0
        # dump_name = "%s_r%d" % (dump_base, rep)
        # print(dump_name)
        # with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, inc_states) = pk.load(f)
        # patterns, wildcards, macros = rules
        # num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        # ### show pdb
        # numrows = min(14, len(macros))
        # numcols = min(15, max(map(len, macros)) + 2)
        # for r in range(numrows):
        #     rule = r if r < numrows/2 else len(patterns)-(numrows-r)
        #     ax = domain.render_subplot(numrows, numcols, r*numcols + 1, patterns[rule])
        #     if r == 0: ax.set_title("pattern")
        #     ax = domain.render_subplot(numrows, numcols, r*numcols + 2, patterns[rule] * (1 - wildcards[rule]))
        #     if r == 0: ax.set_title("trigger")
        #     else: ax.set_title(str(wildcards[rule].sum()))
        #     state = patterns[rule]
        #     for m in range(len(macros[rule])):
        #         if 2+m+1 > numcols: break
        #         state = domain.perform(macros[rule][m], state)
        #         ax = domain.render_subplot(numrows, numcols, r*numcols + 2 + m+1, state)
        #         ax.set_title(str(macros[rule][m]))
        # pt.tight_layout()
        # pt.show()

        # wildcards_disabled = np.zeros(num_incs, dtype=int)
        # rules_added = np.zeros(num_incs, dtype=int)
        # for r in range(len(patterns)):
        #     for w in range(domain.state_size()):
        #         if inc_disabled[r,w] >= len(wildcards_disabled): continue
        #         wildcards_disabled[inc_disabled[r,w]] += 1
        #     rules_added[inc_added[r]] += 1
        # augmented = (rules_added > 0) | (wildcards_disabled > 0)

        # pt.subplot(1,4,1)
        # # pt.plot(np.arange(num_incs), [(inc_added <= i).sum() for i in range(num_incs)])
        # pt.plot(np.arange(num_incs), np.cumsum(rules_added))
        # pt.xlabel("iter")
        # pt.ylabel("num rules")
        # pt.subplot(1,4,2)
        # # pt.plot(np.arange(num_incs), [(inc_disabled[inc_added <= i] > i).sum() for i in range(num_incs)])
        # pt.plot(np.arange(num_incs), np.cumsum(rules_added)*domain.state_size() - np.cumsum(wildcards_disabled))
        # pt.xlabel("iter")
        # pt.ylabel("num wildcards")
        # pt.subplot(1,4,3)
        # pt.plot(np.arange(num_incs), np.cumsum(augmented))
        # pt.plot(np.arange(num_incs), np.cumsum(wildcards_disabled > 0))
        # pt.plot(np.arange(num_incs), np.cumsum(rules_added > 0))
        # pt.legend(["aug", "bad trig", "new rule"])
        # pt.xlabel("iter")
        # pt.ylabel("num augmentations")
        # pt.subplot(1,4,4)
        # # pt.plot(np.arange(num_rules), chain_lengths, 'k.')
        # pt.hist(chain_lengths)
        # pt.xlabel("rule")
        # pt.ylabel("chain length")
        # pt.show()

        # num_problems = 16
        # cats = ["sofar", "recent", "all"]
        # correctness = {cat: list() for cat in cats}
        # godliness = {cat: list() for cat in cats}
        # folkliness = {cat: list() for cat in cats}
        # converge_inc = np.argmax(np.cumsum(augmented))
        # rewind_incs = np.linspace(num_problems, converge_inc, 30).astype(int)
        # # rewind_incs = np.linspace(num_problems, num_incs, 30).astype(int)
        # for rewind_inc in rewind_incs:

        #     rew_patterns, rew_wildcards, rew_macros = rewind(patterns, macros, inc_added, inc_disabled, rewind_inc)
        #     pdb = PatternDatabase(rew_patterns, rew_wildcards, rew_macros, domain)

        #     for cat in cats:

        #         if cat == "sofar": probs = np.random.choice(rewind_inc, size=num_problems) # states so far, up to rewind_inc
        #         if cat == "recent": probs = np.arange(rewind_inc-num_problems, rewind_inc) # moving average near rewind_inc
        #         if cat == "all": probs = np.random.choice(len(all_states), size=num_problems) # all states

        #         num_solved = 0
        #         opt_moves = []
        #         alg_moves = []
            
        #         for p in probs:
        #             state, path = states[inc_states[p]], paths[inc_states[p]]
        #             solved, plan, _, _ = run(state, domain, tree, pdb, max_depth, max_actions, color_neutral)
        #             num_solved += solved            
        #             if solved and len(path) > 0:
        #                 opt_moves.append(len(path))
        #                 alg_moves.append(sum([len(a)+len(m) for _,a,m in plan]))
                
        #         correctness[cat].append( num_solved / num_problems )
        #         godliness[cat].append( np.mean( (np.array(opt_moves) + 1) / (np.array(alg_moves) + 1) ) )
        #         folkliness[cat].append( 1 -  len(rew_macros) / len(all_states) )

        # for c, cat in enumerate(cats):
        #     pt.subplot(1,3, c+1)
        #     pt.plot(rewind_incs, correctness[cat], marker='o', label="correctness")
        #     pt.plot(rewind_incs, godliness[cat], marker='o', label="godliness")
        #     pt.plot(rewind_incs, folkliness[cat], marker='o', label="folkliness")
        #     pt.xlabel("num incs")
        #     pt.ylabel("performance")
        #     pt.ylim([0, 1.1])
        #     pt.legend()
        #     pt.title(cat)
        # pt.show()

        # correctness = {}
        # godliness = {}
        # folkliness = {}

        # num_problems = 32

        # reps = list(range(num_reps))
        # for rep in reps:
        #     print("rep %d..." % rep)

        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, inc_states) = pk.load(f)
        #     patterns, wildcards, macros = rules
        #     num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        #     correctness[rep] = []
        #     godliness[rep] = []
        #     folkliness[rep] = []

        #     rewind_incs = np.linspace(num_problems, num_incs, 32).astype(int)
        #     for rewind_inc in rewind_incs:

        #         rew_patterns, rew_wildcards, rew_macros = rewind(patterns, macros, inc_added, inc_disabled, rewind_inc)
        #         pdb = PatternDatabase(rew_patterns, rew_wildcards, rew_macros, domain)
        #         probs = np.random.choice(len(all_states), size=num_problems) # all states

        #         num_solved = 0
        #         opt_moves = []
        #         alg_moves = []
            
        #         for p in probs:
        #             state, path = all_states[p], optimal_paths[p]
        #             solved, plan, _, _ = run(state, domain, tree, pdb, max_depth, max_actions, color_neutral)
        #             num_solved += solved            
        #             if solved and len(path) > 0:
        #                 opt_moves.append(len(path))
        #                 alg_moves.append(sum([len(a)+len(m) for _,a,m in plan]))
                
        #         correctness[rep].append( num_solved / num_problems )
        #         godliness[rep].append( np.mean( (np.array(opt_moves) + 1) / (np.array(alg_moves) + 1) ) )
        #         # folkliness[rep].append( 1 -  len(rew_macros) / len(all_states) )
        #         folkliness[rep].append( len(rew_macros) )

        # for sp, metric in enumerate(["correctness", "godliness", "folkliness"]):
        #     pt.subplot(1,4, sp+1)
        #     for rep in reps: pt.plot(rewind_incs, locals()[metric][rep], '-', label=str(rep))
        #     pt.ylabel(metric)
        #     pt.xlabel("inc")
        #     # pt.ylim([0, 1.1])
        #     pt.legend()

        # pt.subplot(1,4,4)
        # for rep in reps:
        #     pt.scatter(folkliness[rep][-1], godliness[rep][-1], color='k')
        # pt.xlabel("folkliness")
        # pt.ylabel("godliness")
        # # pt.xlim([0, 1.1])
        # # pt.ylim([0, 1.1])

        # # pt.tight_layout()
        # pt.show()

    # confirm correctness
    if confirm:

        for rep in range(num_reps):

            dump_name = "%s_r%d" % (dump_base, rep)
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, inc_states) = pk.load(f)
            patterns, wildcards, macros = rules
            num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

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

                num_checked += 1
                if verbose and p % (10**min(3, int(np.log10(p+1)))) == 0:
                    print("rep %d: checked %d of %d, %d (%f) solved" % (rep, num_checked, len(all_states), num_solved, num_solved/num_checked))
        
                if not solved and confirm_show:
    
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
        
                # assert solved == domain.is_solved_in(final_state)
                # assert solved
        
            alg_moves = np.array(alg_moves[1:]) # skip solved state from metrics
            opt_moves = np.array(opt_moves[1:])
            alg_opt = alg_moves / opt_moves
            if verbose: print("alg/opt min,max,mean", (alg_opt.min(), alg_opt.max(), alg_opt.mean()))
            if verbose: print("alg min,max,mean", (alg_moves.min(), alg_moves.max(), alg_moves.mean()))
            if verbose: print("Solved %d of %d (%f)" % (num_solved, len(all_states), num_solved/len(all_states)))
            if verbose: print("Rules %d <= %d (%f)" % (len(patterns), len(all_states), len(patterns)/len(all_states)))
    


