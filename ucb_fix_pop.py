import numpy as np
from relaxed_construction import Constructor, rewind
from utils import softmax
from pattern_database import PatternDatabase
from algorithm import run
from approximate_construction import scrambled, uniform
import matplotlib.pyplot as pt

def inc_batch(
    eval_period, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
    constructor, scrambled_objectives, uniform_objectives,
):

    for k in range(eval_period):

        if constructor.num_rules == constructor.max_rules: break

        if inc_sampler == "scrambled": state, path = scrambled_sample()
        if inc_sampler == "uniform": state, path = uniform_sample()

        augmented = constructor.incorporate(state, path)

    if "scrambled" in eval_samplers:
        probs = [scrambled_sample() for _ in range(num_problems)]
        scrambled_objectives.append(constructor.evaluate(probs))

    if "uniform" in eval_samplers:
        probs = [uniform_sample() for _ in range(num_problems)]
        uniform_objectives.append(constructor.evaluate(probs))

    if inc_sampler == "scrambled": correctness, godliness, folkliness = scrambled_objectives[-1]
    if inc_sampler == "uniform": correctness, godliness, folkliness = uniform_objectives[-1]
    
    return correctness, godliness, folkliness

if __name__ == "__main__":

    # config
    cube_size, num_twist_axes, quarter_turns = 2, 2, True # 29k states
    # cube_size, num_twist_axes, quarter_turns = 2, 3, False # 24 states

    tree_depth = 14
    use_safe_depth = False
    max_depth = 1
    max_actions = 30
    color_neutral = False

    num_problems = 32
    eval_period = 1000
    correctness_bar = 1.0

    inc_sampler = "scrambled"
    eval_samplers = ["scrambled", "uniform"]
    assert inc_sampler in eval_samplers

    num_candidates = 16
    max_select_iters = 10000

    breakpoint = -1
    # breakpoint = 100
    num_reps = 10
    break_seconds = 1 * 60
    # break_seconds = 0
    verbose = True

    do_cons = True
    show_results = False
    confirm = False
    confirm_show = False

    # set up descriptive dump name
    dump_period = 1000
    dump_dir = "ufp"
    dump_base = "N%da%dq%d_D%d_M%d_cn%d_%s_P%d" % (
        cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral, inc_sampler, num_candidates)

    import itertools as it
    from cube import CubeDomain
    valid_actions = tuple(it.product(range(num_twist_axes), range(1,cube_size), range(2-quarter_turns, 4, 2-quarter_turns)))
    domain = CubeDomain(cube_size, valid_actions)
    init = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    all_states = tree.states_rooted_at(init)
    optimal_paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved

    max_scramble_length = max_actions - max_depth

    max_rules = len(all_states)

    import pickle as pk
    import os

    if do_cons:

        from time import sleep

        for rep in range(num_reps):

            rng = np.random.default_rng()
            constructors = tuple(
                Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
                for _ in range(num_candidates))

            uniform_sample = uniform(rng, all_states, optimal_paths)
            scrambled_sample = scrambled(domain, rng, max_scramble_length)
            
            scrambled_objectives = tuple([] for _ in range(num_candidates))
            uniform_objectives = tuple([] for _ in range(num_candidates))

            current_objectives = np.empty((num_candidates, 3))
            selection_counts = np.ones(num_candidates)
            exploration = 1.

            # initial evaluation
            print("first evals...")
            for c in range(len(constructors)):
                correctness, godliness, folkliness = inc_batch(
                    eval_period, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
                    constructors[c], scrambled_objectives[c], uniform_objectives[c],
                )
                current_objectives[c] = correctness, godliness, folkliness
                if verbose:
                    wildcards = constructors[c].rules()[1]
                    print("%d init: %d <= %d rules (%d states), %f solved, %f godly, %f wildcard" % (
                        c, constructors[c].num_rules, max_rules, len(all_states),
                        correctness, godliness, wildcards.sum() / wildcards.size))

            # ucb selection
            for select_iter in range(1,max_select_iters):
                
                # Q = current_objectives[:,0] # correctness
                Q = current_objectives[:,1] # godliness
                # Q = current_objectives[:,2] # folkliness
                ucb_logits = Q + exploration * np.sqrt(np.log(select_iter) / (selection_counts+1))
                c = ucb_logits.argmax() # hard

                selection_counts[c] += 1

                correctness, godliness, folkliness = inc_batch(
                    eval_period, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
                    constructors[c], scrambled_objectives[c], uniform_objectives[c],
                )
                current_objectives[c] = correctness, godliness, folkliness

                if verbose:
                    wildcards = constructors[c].rules()[1]
                    print("%d,%d: %d selects, %d <= %d rules (%d states), %f solved, %f godly, %f wildcard" % (
                        select_iter, c, selection_counts[c], constructors[c].num_rules, max_rules, len(all_states),
                        correctness, godliness, wildcards.sum() / wildcards.size))

                dump_name = "%s_r%d" % (dump_base, rep)
                rules = [constructors[c].rules() for c in range(num_candidates)]
                logs = [constructors[c].logs() for c in range(num_candidates)]
                with open(dump_name + ".pkl", "wb") as f:
                    pk.dump((rules, logs, (scrambled_objectives, uniform_objectives)), f)

                if correctness >= correctness_bar: break        
    
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            rules = [constructors[c].rules() for c in range(num_candidates)]
            logs = [constructors[c].logs() for c in range(num_candidates)]
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((rules, logs, (scrambled_objectives, uniform_objectives)), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))
    
            if verbose: print("Breaking for %s seconds..." % str(break_seconds))
            sleep(break_seconds)

    if show_results:

        reps = list(range(3))
        for rep in reps:
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f: (rules, logs, objectives) = pk.load(f)
            scrambled_objectives, uniform_objectives = objectives
    
            for c in range(num_candidates):
                print(c)
                patterns, wildcards, macros = rules[c]
                num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs[c]
                
                # incs = np.arange(0, num_incs, eval_period)
                correctness, godliness, folkliness = zip(*scrambled_objectives[c])
                
                pt.subplot(len(reps), 3, rep*3 + 1)
                pt.plot(correctness)
                pt.xlabel("evals")
                pt.ylabel("correctness")
                pt.subplot(len(reps), 3, rep*3 + 2)
                pt.plot(godliness)
                pt.xlabel("evals")
                pt.ylabel("godliness")
                pt.subplot(len(reps), 3, rep*3 + 3)
                pt.plot(folkliness)
                pt.xlabel("evals")
                pt.ylabel("folkliness")

        pt.show()

