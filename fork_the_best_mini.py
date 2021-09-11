import numpy as np
import matplotlib.pyplot as pt
import pickle as pk
import os
from time import sleep, perf_counter
import itertools as it
from relaxed_construction import Constructor, rewind
from approximate_construction import scrambled, uniform

class Trail:
    def __init__(self, constructor, max_incs, sample, decay_rate, exhaust_period, all_probs, correctness_bar):
        self.constructor = constructor
        self.max_incs = max_incs
        self.sample = sample
        self.decay_rate = decay_rate
        self.exhaust_period = exhaust_period
        self.all_probs = all_probs
        self.correctness_bar = correctness_bar
        self.true_correctness = [0.]
        self.ema_correctness = [0.]

    def rewind_copy(self, inc):
        trail = Trail(self.constructor, self.max_incs, self.sample, self.decay_rate, self.exhaust_period, self.all_probs, self.correctness_bar)
        trail.constructor = self.constructor.rewind_copy(inc)
        trail.true_correctness = list(self.true_correctness[:(inc // self.exhaust_period)+1])
        trail.ema_correctness = list(self.ema_correctness[:inc+1])
        return trail

    def run_to_convergence(self, verbose_prefix=None):
        for k in it.count():
            if self.constructor.num_rules == self.constructor.max_rules: break
            if self.constructor.num_incs == self.max_incs: break
    
            state, path = self.sample()
            augmented = self.constructor.incorporate(state, path)
            self.ema_correctness.append(self.decay_rate * self.ema_correctness[-1] + (1 - self.decay_rate) * (1 - augmented))
    
            if self.constructor.num_incs % self.exhaust_period == 0:
                true_correctness, _, _ = self.constructor.evaluate(self.all_probs)
                self.true_correctness.append(true_correctness)
    
            if verbose_prefix != None:
                print("%s%d: %f(%f)" % (verbose_prefix, k, self.ema_correctness[-1], self.true_correctness[-1]))
    
            if self.ema_correctness[-1] > self.correctness_bar: break
            if self.true_correctness[-1] == 1.0: break

if __name__ == "__main__":

    # config
    do_cons = False
    show_results = True
    confirm = True
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

    num_forks = 500
    num_eval_problems = 32
    exhaust_period = 100

    decay_rate = .99
    correctness_bar = 1.1
    max_incs = 50000
    inc_sampler = "scrambled"

    num_reps = 100
    break_seconds = 0 * 60
    verbose = True

    # set up descriptive dump name
    dump_period = 10000
    dump_dir = "ftb"
    dump_base = "N%d%s_D%d_M%d_cn%d_%s_P%d_e%s" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral, inc_sampler, num_eval_problems, decay_rate)
    print(dump_base)

    from cube import CubeDomain
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    all_states = tree.states_rooted_at(init)
    optimal_paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved

    max_scramble_length = max_actions - max_depth
    max_rules = len(all_states)

    rng = np.random.default_rng()
    if inc_sampler == "uniform": sample = uniform(rng, all_states, optimal_paths)
    if inc_sampler == "scrambled": sample = scrambled(domain, rng, max_scramble_length)

    # if inc_sampler == "uniform": all_probs = list(zip(all_states, optimal_paths))
    # if inc_sampler == "scrambled": all_probs = [sample() for _ in range(3*len(all_states))]
    all_probs = list(zip(all_states, optimal_paths))

    if do_cons:

        for rep in range(num_reps):
            rep_start = perf_counter()

            # # initial scalarization weights in S+
            weights = rng.normal(size=2) # random
            # weights = np.array([0.01, .99]) # folksy
            # weights = np.array([.99, .01]) # godly
            # weights = np.array([.5, .5]) # equally important
            weights = np.fabs(weights)
            weights /= np.sqrt(np.sum(weights**2))

            def scalarize(objs):
                return np.maximum(0, np.array(objs) / weights).min(axis=-1) ** 2

            def run_and_eval(trail, verbose_prefix):
                trail.run_to_convergence(verbose_prefix)
                probs = [sample() for _ in range(num_eval_problems)]
                sample_objectives = trail.constructor.evaluate(probs)
                exhaust_objectives = trail.constructor.evaluate(trail.all_probs)
                sample_scalarized = scalarize(sample_objectives[1:])
                exhaust_scalarized = scalarize(exhaust_objectives[1:])
                return sample_scalarized, exhaust_scalarized, sample_objectives, exhaust_objectives

            # run first trail to completion
            constructor = Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
            trail = Trail(constructor, max_incs, sample, decay_rate, exhaust_period, all_probs, correctness_bar)
            evals = run_and_eval(trail, "Rep %d Initial trail: " % rep if verbose else None)
            best_index, fork_inc = 0, 0
            leaves = [(evals, best_index, fork_inc, trail.constructor.num_incs)]

            num_dominated = 0
            for fork in range(num_forks):
                if len(trail.constructor.augment_incs) - num_dominated - 2 < 0: break

                rewind_index = len(trail.constructor.augment_incs) - num_dominated - 2
                fork_inc = trail.constructor.augment_incs[rewind_index]
                child = trail.rewind_copy(fork_inc)
                
                # evals = run_and_eval(child, verbose_prefix="  ")
                evals = run_and_eval(child, verbose_prefix=None)
                leaves.append((evals, best_index, fork_inc, child.constructor.num_incs))

                if verbose:
                    print("Rep %d, fork %d: best leaf %d gets %f(%f), dominated %d (%d augments), rewinding %d to %d gets %f(%f)" % (
                        rep, fork, best_index,
                        leaves[best_index][0][0], leaves[best_index][0][1],
                        num_dominated, len(trail.constructor.augment_incs),
                        trail.constructor.augment_incs[-1], fork_inc,
                        leaves[-1][0][0], leaves[-1][0][1],
                    ))

                sample_scalarized = evals[0]
                if sample_scalarized < leaves[best_index][0][0]:
                    num_dominated += 1
                elif child.constructor.num_incs < trail.constructor.num_incs or sample_scalarized > leaves[best_index][0][0]:
                    num_dominated = 0
                    trail = child
                    best_index = len(leaves)-1

            print("exhaust obj:")
            print(leaves[best_index][0][3])

            dump_name = "%s_r%d" % (dump_base, rep)
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((trail.constructor.rules(), trail.constructor.logs(), weights, leaves), f)
            os.system("mv %s_r%d.pkl %s/" % (dump_base, rep, dump_dir))

    if show_results:

        from matplotlib import rcParams
        # rcParams['font.family'] = 'sans-serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 12
        rcParams['text.usetex'] = True

        rep = 0
        dump_name = "%s_r%d" % (dump_base, rep)
        dump_path = "%s/%s.pkl" % (dump_dir, dump_name)
        with open(dump_path, "rb") as f:
            (rules, logs, weights, leaves) = pk.load(f)
        print("weights:", *weights)

        evals, parent_index, fork_inc, num_incs = zip(*leaves)
        sample_scalarized, exhaust_scalarized, _, _ = zip(*evals)
        
        pt.figure(figsize=(3.5, 5))
        for sp, name in enumerate(["Sample", "Ground truth"]):
            pt.subplot(3,1,sp+1)
            pt.title(name)

            scalarized = [sample_scalarized, exhaust_scalarized][sp]
            for n in range(len(leaves)):
                i = parent_index[n]
                while fork_inc[n] < fork_inc[i]: i = parent_index[i]
                pt.plot(
                    [fork_inc[i], fork_inc[i], num_incs[n]],
                    [scalarized[i], scalarized[n], scalarized[n]],
                    '-', color=(.75 - .5 * n / len(leaves),)*3)
                    # '-', color=(.75 * n / len(leaves),)*3)
                    # '-', color=(.5,)*3)

            n = np.argmax(scalarized)
            inc = num_incs[n]
            while True:
                i = parent_index[n]
                while fork_inc[n] < fork_inc[i]: i = parent_index[i]
                pt.plot(
                    [fork_inc[i], fork_inc[i], inc],
                    [scalarized[i], scalarized[n], scalarized[n]],
                    '-', color=(.0,)*3)
                if n == 0: break
                n = i
                inc = fork_inc[i]

            pt.xlabel("Number of modifications")
            pt.ylabel("Scalarization value")

        pt.subplot(3,1,3)
        sampled = [leaf[0][0] for leaf in leaves]
        exhaust = [leaf[0][1] for leaf in leaves]
        pt.scatter(sampled, exhaust, color='k')
        pt.xlabel("Sample")
        pt.ylabel("Ground truth")
        pt.title("Scalarization values")
        pt.tight_layout()
        pt.savefig("ftb_%s.pdf" % dump_name)
        # pt.show()
        pt.close()

        pt.figure(figsize=(3.5, 3.25))
        for sam,sampler in enumerate(["scrambled", "uniform"]):

            dump_base = "N%d%s_D%d_M%d_cn%d_%s_P%d_e%s" % (
                cube_size, cube_str, tree_depth, max_depth, color_neutral, sampler, num_eval_problems, decay_rate)

            most_folksy, folksy_rules, folksy_rep = None, None, None
            for rep in range(num_reps):
                dump_name = "%s_r%d" % (dump_base, rep)
                dump_path = "%s/%s.pkl" % (dump_dir, dump_name)
                with open(dump_path, "rb") as f:
                    (rules, logs, weights, leaves) = pk.load(f)
                print(rep, "weights:", *weights)
        
                evals, parent_index, fork_inc, num_incs = zip(*leaves)
                sample_scalarized, exhaust_scalarized, sample_objectives, exhaust_objectives = zip(*evals)
        
                # for sp, name in enumerate(["Samp.", "Pop."]):
                for sp, name in enumerate([sampler, "population"]):
                    pt.subplot(2,2,2*sam + sp+1)
    
                    scalarized = [sample_scalarized, exhaust_scalarized][sp]
                    best = np.argmax(scalarized)
                    objectives = [sample_objectives, exhaust_objectives][sp]
                    _, godliness, folkliness = zip(*objectives)
                    if sam == 1: pt.xlabel("Folksiness")
                    if sp == 0: pt.ylabel("Godliness")
    
                    if sp == 0 and (most_folksy == None or folkliness[best] > most_folksy):
                        most_folksy = folkliness[best]
                        fewest_rules = (1 - most_folksy) * max_rules
                        folksy_rules = rules
                        folksy_rep = rep
    
                    # from constructor:
                    # godliness = mean(0 if not solved else 1 / max(alg_moves,1)) # just try to take smaller step counts on average
                    # folkliness = 1 - self.num_rules / self.max_rules
                    godliness = 1 / np.array(godliness) # num steps
                    folkliness = (1 - np.array(folkliness)) * max_rules # num rules
                    if sam == 1: pt.xlabel("Number of rules")
                    if sp == 0: pt.ylabel("Avg. soln. len.")
    
                    # pt.scatter(folkliness[::20], godliness[::20], color=(.5,)*3, zorder=1)
                    # pt.scatter(folkliness[best], godliness[best], color=(.0,)*3, zorder=2)
                    pt.plot(folkliness[::50], godliness[::50], '.', color=(.5,)*3, zorder=1)
                    pt.plot(folkliness[best], godliness[best], '.', color=(.0,)*3, zorder=2)
                    if sam == 0: pt.ylim([0, 25])
                    if sam == 1: pt.ylim([0, 15])
                    # if sam == 0: pt.title(name)
                    # pt.title("%s (%s)" % (name, sampler))
                    pt.title(name, fontsize=12)

        print("most folksy", most_folksy)
        print("fewest rules", fewest_rules)
    
        pt.tight_layout()
        pt.savefig("ftb_s120_pareto.pdf")
        # pt.show()
        pt.close()

        # pt.figure(figsize=(3.5, 2))
        # most_folksy, folksy_rules = None, None
        # for rep in range(num_reps):
        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     dump_path = "%s/%s.pkl" % (dump_dir, dump_name)
        #     with open(dump_path, "rb") as f:
        #         (rules, logs, weights, leaves) = pk.load(f)
        #     print(rep, "weights:", *weights)
    
        #     evals, parent_index, fork_inc, num_incs = zip(*leaves)
        #     sample_scalarized, exhaust_scalarized, sample_objectives, exhaust_objectives = zip(*evals)
    
        #     for sp, name in enumerate(["Sample", "Population"]):
        #         pt.subplot(1,2,sp+1)

        #         scalarized = [sample_scalarized, exhaust_scalarized][sp]
        #         best = np.argmax(scalarized)
        #         objectives = [sample_objectives, exhaust_objectives][sp]
        #         _, godliness, folkliness = zip(*objectives)
        #         pt.xlabel("Folksiness")
        #         if sp == 0: pt.ylabel("Godliness")

        #         if sp == 0 and (most_folksy == None or folkliness[best] > most_folksy):
        #             most_folksy = folkliness[best]
        #             folksy_rules = rules

        #         # from constructor:
        #         # godliness = mean(0 if not solved else 1 / max(alg_moves,1)) # just try to take smaller step counts on average
        #         # folkliness = 1 - self.num_rules / self.max_rules
        #         godliness = 1 / np.array(godliness) # num steps
        #         folkliness = (1 - np.array(folkliness)) * max_rules # num rules
        #         pt.xlabel("Number of rules")
        #         if sp == 0: pt.ylabel("Avg. soln. length")

        #         pt.scatter(folkliness[::10], godliness[::10], color=(.5,)*3, zorder=1)
        #         pt.scatter(folkliness[best], godliness[best], color=(.0,)*3, zorder=2)
        #         pt.ylim([0, 25])
        #         pt.title(name)
    
        # pt.tight_layout()
        # pt.savefig("ftb_s120_pareto.pdf")
        # pt.show()
        # pt.close()

        ### show folksy pdb
        # pt.figure(figsize=(15, 10))
        # patterns, wildcards, macros = folksy_rules
        # numrows = min(7, len(macros))
        # numcols = min(15, max(map(len, macros)) + 3)
        # for r in range(numrows):
        #     rule = r if r < numrows/2 else len(patterns)-(numrows-r)
        #     ax = domain.render_subplot(numrows, numcols, r*numcols + 1, patterns[rule])
        #     if r == 0: ax.set_title("pattern")
        #     ax = domain.render_subplot(numrows, numcols, r*numcols + 2, patterns[rule] * (1 - wildcards[rule]))
        #     if r == 0: ax.set_title("trigger")
        #     else: ax.set_title(str(wildcards[rule].sum()))
        #     ax = domain.render_subplot(numrows, numcols, r*numcols + 3, domain.orientations_of(patterns[rule] * (1 - wildcards[rule]))[7])
        #     if r == 0: ax.set_title("behind")
        #     state = patterns[rule]
        #     for m in range(len(macros[rule])):
        #         if 3+m+1 > numcols: break
        #         state = domain.perform(macros[rule][m], state)
        #         ax = domain.render_subplot(numrows, numcols, r*numcols + 3 + m+1, state)
        #         ax.set_title(str(macros[rule][m]))
        # # pt.tight_layout()
        # pt.show()
        # pt.close()

        def fnb(numrows, numcols, sp, state):
            ax = domain.render_subplot(numrows, numcols, sp, state, x0=-3, txt=False)
            ax = domain.render_subplot(numrows, numcols, sp, domain.orientations_of(state)[7], x0=3, txt=False)
            return ax

        pt.figure(figsize=(6.5, 9))
        patterns, wildcards, macros = folksy_rules
        # numrows = min(15, max(map(len, macros)) + 3)
        # numcols = min(7, len(macros))
        numrows = max(map(len, macros)) + 2
        numcols = len(macros)
        for rule in range(numcols):
            ax = fnb(numrows, numcols, 0*numcols + rule + 1, patterns[rule])
            ax.set_title(r"$S_{%d}$" % rule)
            trigger = patterns[rule] * (1 - wildcards[rule])
            ax = fnb(numrows, numcols, 1*numcols + rule + 1, trigger)
            ax.set_title(r"$S_{%d} \vee W_{%d}$" % (rule, rule))
            state = trigger
            for m in range(len(macros[rule])):
                state = domain.perform(macros[rule][m], state)
                ax = fnb(numrows, numcols, (m+2)*numcols + rule + 1, state)
                ax.set_title(str(macros[rule][m]))
        pt.tight_layout()
        pt.savefig("folksy_rules.pdf")
        pt.show()
        pt.close()

        # confirm folksy correctness
        if confirm:

            from pattern_database import PatternDatabase
            from algorithm import run

            rep = folksy_rep
            pdb = PatternDatabase(patterns, wildcards, macros, domain)    
            num_checked = 0
            num_solved = 0
            opt_moves = []
            alg_moves = []

            for p, (prob_state, path) in enumerate(all_probs):

                solved, plan, rule_indices, interstates = run(
                    prob_state, domain, tree, pdb, max_depth, max_actions, color_neutral)
                print(rule_indices)
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

