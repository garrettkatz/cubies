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
    do_cons = False
    show_results = True
    confirm = False
    confirm_show = False

    # pocket cube: one axis with quarter twists, one with half twists
    # 120 states, max depth 11
    cube_size = 2
    valid_actions = (
        (0,1,1), (0,1,2), (0,1,3),
        (1,1,2), 
    )
    cube_str = "s120"
    tree_depth = 11

    # # pocket cube: one axis with quarter twists, two with half twists
    # # 5040 states, reached in max_depth=13
    # cube_size = 2
    # valid_actions = (
    #     (0,1,1), (0,1,2), (0,1,3),
    #     (1,1,2),
    #     (2,1,2),
    # )
    # cube_str = "s5040"
    # tree_depth = 13

    use_safe_depth = False
    max_depth = 1
    max_actions = 30
    color_neutral = False

    # static_incs_for_stop = 256
    num_problems = 32
    eval_period = 1
    correctness_bar = 1.1
    gamma = .99
    inc_sampler = "uniform"
    eval_samplers = ["scrambled", "uniform"]
    # eval_samplers = ["scrambled"]
    assert inc_sampler in eval_samplers

    breakpoint = -1
    # breakpoint = 100
    num_reps = 30
    break_seconds = 0 * 60
    verbose = True

    # set up descriptive dump name
    dump_period = 100
    dump_dir = "acons"
    dump_base = "N%d%s_D%d_M%d_cn%d_%s_cb%s" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral, inc_sampler, correctness_bar)

    import itertools as it
    from cube import CubeDomain
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    all_states = tree.states_rooted_at(init)
    optimal_paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved
    max_rules = len(all_states)

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
            exhaust_objectives = []
            inc_objectives = []
            static_incs = [0]

            for k in it.count():
                if constructor.num_rules in [max_rules, breakpoint]: break

                if inc_sampler == "scrambled": state, path = scrambled_sample()
                if inc_sampler == "uniform": state, path = uniform_sample()
                # state, path = all_states[k % len(all_states)], optimal_paths[k % len(all_states)]

                state_bytes = state.tobytes()
                if state_bytes not in inc_states: inc_states[state_bytes] = []
                inc_states[state_bytes].append(k)

                probs = [(state, path)]
                correctness, godliness, folkliness = constructor.evaluate(probs)
                inc_objectives.append((correctness, godliness, folkliness))

                augmented = constructor.incorporate(state, path)
                if augmented:
                    static_incs.append(0)
                else:
                    static_incs.append(static_incs[-1] + 1)

                if k % eval_period == 0:

                    if "scrambled" in eval_samplers:
                        probs = [scrambled_sample() for _ in range(num_problems)]
                        scrambled_objectives.append(constructor.evaluate(probs))

                    if "uniform" in eval_samplers:
                        probs = [uniform_sample() for _ in range(num_problems)]
                        uniform_objectives.append(constructor.evaluate(probs))

                    # if inc_sampler == "scrambled": probs = [scrambled_sample() for _ in range(len(all_states))]
                    # if inc_sampler == "uniform": probs = list(zip(all_states, optimal_paths))
                    probs = list(zip(all_states, optimal_paths))
                    exhaust_objectives.append(constructor.evaluate(probs))

                    if inc_sampler == "scrambled": correctness, godliness, _ = scrambled_objectives[-1]
                    if inc_sampler == "uniform": correctness, godliness, _ = uniform_objectives[-1]
                    true_correctness, _, _ = exhaust_objectives[-1]

                    if verbose:
                        wildcards = constructor.rules()[1]
                        print("%d,%d: %d <= %d rules (%d states), %f(%f) solved, %f godly, %f wildcard, static for %d" % (
                            rep, k, constructor.num_rules, max_rules, len(all_states),
                            correctness, true_correctness, godliness,
                            wildcards.sum() / wildcards.size, static_incs[-1]))

                    if correctness >= correctness_bar: break
                    
                    if true_correctness == 1.0: break

                if k % dump_period == 0:
                    dump_name = "%s_r%d" % (dump_base, rep)
                    with open(dump_name + ".pkl", "wb") as f:
                        pk.dump((
                            constructor.rules(), constructor.logs(),
                            (scrambled_objectives, uniform_objectives, exhaust_objectives, inc_objectives),
                            static_incs, inc_states), f)
        
            if verbose: print("(max_depth = %d)" % max_depth)
    
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((
                    constructor.rules(), constructor.logs(),
                    (scrambled_objectives, uniform_objectives, exhaust_objectives, inc_objectives),
                    static_incs, inc_states), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))
    
            # patterns, wildcards, macros = constructor.rules()
            # np.set_printoptions(linewidth=200)
            # for k in range(10): print(patterns[k])
            # for k in range(10): print(patterns[-k])

            if verbose: print("Breaking for %s seconds..." % str(break_seconds))
            sleep(break_seconds)

    if show_results:

        from matplotlib import rcParams
        # rcParams['font.family'] = 'sans-serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 9

        # rep = 0
        # dump_name = "%s_r%d" % (dump_base, rep)
        # # with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, static_incs, inc_states) = pk.load(f)
        # with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, inc_states) = pk.load(f)
        # patterns, wildcards, macros = rules

        # ### show pdb
        # numrows = min(3, len(macros))
        # numcols = min(4, max(map(len, macros)) + 2)
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
        # # pt.tight_layout()
        # pt.show()

        # show one cons run
        rep = 0
        dump_name = "%s_r%d" % (dump_base, rep)
        print(dump_name)
        with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, static_incs, inc_states) = pk.load(f)
        num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        if inc_sampler == "scrambled":
            sampled_correct, sampled_godliness, folkliness = list(zip(*objectives[0]))
        if inc_sampler == "uniform":
            sampled_correct, sampled_godliness, folkliness = list(zip(*objectives[1]))
        true_correct, true_godliness, _ = list(zip(*objectives[2]))
        inc_correct, inc_godliness, _ = list(zip(*objectives[3]))
        inc_correct, inc_godliness = map(list, (inc_correct, inc_godliness))
        
        # EMA
        # inc_correct[0] = 0
        # inc_godliness[0] = 0
        for k in range(1, len(inc_correct)):
            inc_correct[k] = gamma * inc_correct[k-1] + (1 - gamma) * inc_correct[k]
            inc_godliness[k] = gamma * inc_godliness[k-1] + (1 - gamma) * inc_godliness[k]
            # if static_incs[k] > 0:
            #     inc_correct[k] = gamma * inc_correct[k-1] + (1 - gamma) * inc_correct[k]
            #     inc_godliness[k] = gamma * inc_godliness[k-1] + (1 - gamma) * inc_godliness[k]
            # else:
            #     inc_correct[k] = 0
            #     inc_godliness[k] = 0
            #     pass
        # # online mean since last augmentation
        # for k in range(1, len(inc_correct)):
        #     if static_incs[k] > 0:
        #         inc_correct[k] = inc_correct[k-1] + (inc_correct[k] - inc_correct[k-1]) / (static_incs[k] + 1)
        #         inc_godliness[k] = inc_godliness[k-1] + (inc_godliness[k] - inc_godliness[k-1]) / (static_incs[k] + 1)
        #     else:
        #         inc_correct[k] = 0
        #         inc_godliness[k] = 0
        #         pass
                

        # pt.subplot(4,1,1)
        # pt.plot(static_incs, 'k-')
        # pt.ylabel("Consecutive static")
        # pt.subplot(4,1,2)
        # pt.plot(folkliness, 'k-')
        # pt.ylabel("Folkliness")
        # pt.subplot(4,1,3)
        # # pt.plot(sampled_correct, '-', color=(.5,)*3, label="Sample")
        # pt.plot(inc_correct, '-', color=(.5,)*3, label="Online")
        # pt.plot(true_correct, '-', color=(0,)*3, label="Population")
        # pt.legend()
        # pt.ylabel("Correctness")
        # pt.subplot(4,1,4)
        # # pt.plot(sampled_godliness, '-', color=(.5,)*3, label="Sample")
        # pt.plot(inc_godliness, '-', color=(.5,)*3, label="Online")
        # pt.plot(true_godliness, '-', color=(0,)*3, label="Population")
        # pt.legend()
        # pt.ylabel("Godliness")
        # pt.xlabel("Number of incorporations")
        # pt.tight_layout()
        # pt.show()

        pt.figure(figsize=(3.5, 2.5))
        pt.subplot(2,1,1)
        pt.plot(static_incs, 'k-')
        pt.ylabel("Static Window")
        pt.subplot(2,1,2)
        pt.plot(true_correct, '-', color=(.75,)*3, label="Ground truth")
        pt.plot(inc_correct, '-', color=(0,)*3, label="EMA")
        # if inc_sampler == "scrambled":
        #     pt.plot(np.arange(0,len(sampled_correct),10), sampled_correct[::10], '--', color=(0,)*3, label="Random sample")
        pt.legend()
        pt.ylabel("Correctness")
        pt.xlabel("Number of incorporations")
        pt.tight_layout()
        pt.savefig("acons_%s.pdf" % dump_name)
        pt.show()
        pt.close()

        pt.figure(figsize=(3.5, 1.5))
        data = [[],[]]
        for s, sampler in enumerate(["uniform", "scrambled"]):
            for rep in range(num_reps):

                fname = "N%d%s_D%d_M%d_cn%d_%s_cb%s_r%d" % (
                    cube_size, cube_str, tree_depth, max_depth, color_neutral, sampler, correctness_bar, rep)
                print(fname)
                if not os.path.exists("%s/%s.pkl" % (dump_dir, fname)): break
                with open("%s/%s.pkl" % (dump_dir, fname), "rb") as f: (rules, logs, objectives, static_incs, inc_states) = pk.load(f)
                num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs
                data[s].append(num_incs)

        # pt.hist(data, color=[(1,)*3, (.5,)*3], ec='k')
        pt.hist(data[0], bins = np.arange(0,max(data[0]),200), color=(.5,)*3, ec='k', rwidth=.75, align="left")
        pt.hist(data[1], bins = np.arange(0,max(data[1]),200), color=(1,)*3, ec='k', rwidth=.75, align="right")
        pt.xlabel("Iterations to convergence")
        pt.ylabel("Frequency")
        pt.legend(["Scrambled","Uniform"])
        pt.tight_layout()
        pt.savefig("acons_%s_hist.pdf" % cube_str)
        pt.show()

        # # for rep in range(num_reps):
        # for rep in range(3):
        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     print(dump_name)
        #     if not os.path.exists("%s/%s.pkl" % (dump_dir, dump_name)): break
        #     with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives, static_incs, inc_states) = pk.load(f)
        #     num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs

        #     pt.plot(static_incs)
        #     pt.xlabel("incs")
        #     pt.xlabel("incs since augment")
        #     pt.show()

        #     true_correct = list(zip(*objectives[2]))[0]

        #     incs = np.arange(0, num_incs, eval_period)
        #     for n, (name, objective) in enumerate(zip(["scrambled","uniform"], objectives[:2])):
        #         if len(objective) == 0: break
        #         correctness, godliness, folkliness = zip(*objective)
        #         correctness = list(correctness)
        #         godliness = list(godliness)
        #         for i in range(1,len(correctness)):
        #             # correctness[i] = .9*correctness[i-1] + .1 * correctness[i]
        #             godliness[i] = .9*godliness[i-1] + .1 * godliness[i]
        #         pt.subplot(2,3,n*3 + 1)
        #         pt.plot(incs, correctness, label="sampled")
        #         pt.plot(incs, true_correct, label="true")
        #         pt.ylabel("correctness")
        #         pt.legend()
        #         pt.title(name)
        #         pt.xlabel("incs")
        #         pt.subplot(2,3,n*3 + 2)
        #         pt.plot(incs, godliness)
        #         pt.ylabel("godliness")
        #         pt.title(name)
        #         pt.xlabel("incs")
        #         pt.subplot(2,3,n*3 + 3)
        #         pt.plot(incs, folkliness)
        #         pt.ylabel("folkliness")
        #         pt.title(name)
        #         pt.xlabel("incs")
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
    


