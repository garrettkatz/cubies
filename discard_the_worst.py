import numpy as np
from relaxed_construction import Constructor, rewind
from utils import softmax
from pattern_database import PatternDatabase
from algorithm import run
from approximate_construction import scrambled, uniform
import matplotlib.pyplot as pt

def inc_batch(
    batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
    constructor, scrambled_objectives, uniform_objectives,
):

    for k in range(batch_size):

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
    batch_size = 32
    correctness_bar = 1.0

    inc_sampler = "scrambled"
    eval_samplers = ["scrambled", "uniform"]
    assert inc_sampler in eval_samplers

    num_candidates = 16
    max_batch_iters = 10000
    ema_factor = .9
    keep_one = int(True)

    breakpoint = -1
    # breakpoint = 100
    num_reps = 20
    break_seconds = 1 * 60
    # break_seconds = 0
    verbose = True

    do_cons = True
    show_results = False
    confirm = False
    confirm_show = False

    # set up descriptive dump name
    dump_period = 10
    dump_dir = "dtw"
    dump_base = "N%da%dq%d_D%d_M%d_cn%d_%s_P%d_B%d_e%s" % (
        cube_size, num_twist_axes, quarter_turns, tree_depth, max_depth, color_neutral, inc_sampler, num_candidates, batch_size, ema_factor)

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

        from time import sleep, perf_counter

        for rep in range(num_reps):

            rep_start = perf_counter()

            rng = np.random.default_rng()
            constructors = list(
                Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
                for _ in range(num_candidates))

            uniform_sample = uniform(rng, all_states, optimal_paths)
            scrambled_sample = scrambled(domain, rng, max_scramble_length)
            
            scrambled_objectives = tuple([] for _ in range(num_candidates))
            uniform_objectives = tuple([] for _ in range(num_candidates))

            replacements = []
            ema_objectives = []
            current_objectives = np.empty((num_candidates, 3))

            batch_times = []

            for batch_iter in range(max_batch_iters):

                batch_start = perf_counter()

                # evaluate each candidate
                for c in range(num_candidates):
                    correctness, godliness, folkliness = inc_batch(
                        batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
                        constructors[c], scrambled_objectives[c], uniform_objectives[c],
                    )
                    if batch_iter == 0:
                        current_objectives[c] = correctness, godliness, folkliness
                    else:
                        current_objectives[c,:2] *= ema_factor
                        current_objectives[c,:2] += (1 - ema_factor) * np.array([correctness, godliness])
                        current_objectives[c,2] = folkliness

                ema_objectives.append(current_objectives.copy())

                # replace the worst with copy of the best
                worst = np.argmin(current_objectives[keep_one:,2]) + keep_one # folkliness
                best = np.argmax(current_objectives[keep_one:,2]) + keep_one # folkliness
                replacements.append((best, worst))

                if verbose:
                    print("%d: %d [%d r, ~%f s, ~%f g, ~%f f] --> %d [%d rules, ~%f s, ~%f g, ~%f f] cf [%d rules, ~%f s, ~%f g, ~%f f]" % (
                        batch_iter,
                        worst, constructors[worst].num_rules, current_objectives[worst,0], current_objectives[worst,1], current_objectives[worst,2],
                        best, constructors[best].num_rules, current_objectives[best,0], current_objectives[best,1], current_objectives[best,2],
                        constructors[0].num_rules, current_objectives[0,0], current_objectives[0,1], current_objectives[0,2],
                    ))

                constructors[worst] = constructors[best].copy()
                current_objectives[worst] = current_objectives[best]

                batch_times.append(perf_counter() - batch_start)

                if batch_iter % dump_period == 0:
                    dump_name = "%s_r%d" % (dump_base, rep)
                    rules = [constructors[c].rules() for c in range(num_candidates)]
                    logs = [constructors[c].logs() for c in range(num_candidates)]
                    choices = [constructors[c].choices() for c in range(num_candidates)]
                    with open(dump_name + ".pkl", "wb") as f:
                        pk.dump((rules, logs, choices, (scrambled_objectives, uniform_objectives), ema_objectives, replacements, batch_times), f)

                if correctness >= correctness_bar: break        

            rep_time = perf_counter() - rep_start

            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            rules = [constructors[c].rules() for c in range(num_candidates)]
            logs = [constructors[c].logs() for c in range(num_candidates)]
            choices = [constructors[c].choices() for c in range(num_candidates)]
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((rules, logs, choices, (scrambled_objectives, uniform_objectives), ema_objectives, replacements, batch_times), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))

            if verbose:
                print("Took %s seconds total." % rep_time)
                print("Breaking for %s seconds..." % str(break_seconds))

            sleep(break_seconds)

    if show_results:

        reps = list(range(3))
        # reps = list(range(num_reps))
        for rep in reps:
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
                (rules, logs, choices, objectives, ema_objectives, replacements, batch_times) = pk.load(f)
            scrambled_objectives, uniform_objectives = objectives
            ema_objectives = np.stack(ema_objectives)
    
            for c in range(num_candidates):
                print(c)
                patterns, wildcards, macros = rules[c]
                num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs[c]
                
                # correctness, godliness, folkliness = zip(*scrambled_objectives[c])
                correctness, godliness, folkliness = ema_objectives[:,:,0], ema_objectives[:,:,1], ema_objectives[:,:,2]
                folkliness = np.array([(inc_added < i).sum() for i in range(0,num_incs, batch_size)])
                
                pt.subplot(len(reps), 3, rep*3 + 1)
                pt.plot(correctness)
                pt.xlabel("batches")
                pt.ylabel("correctness")
                pt.subplot(len(reps), 3, rep*3 + 2)
                pt.plot(godliness)
                pt.xlabel("batches")
                pt.ylabel("godliness")
                pt.subplot(len(reps), 3, rep*3 + 3)
                pt.plot(folkliness)
                pt.xlabel("batches")
                pt.ylabel("folkliness")

        pt.show()

        reps = list(range(num_reps))
        for rep in reps:
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            if not os.path.exists("%s/%s.pkl" % (dump_dir, dump_name)): break
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
                (rules, logs, choices, objectives, ema_objectives, replacements, batch_times) = pk.load(f)
            scrambled_objectives, uniform_objectives = objectives
            ema_objectives = np.stack(ema_objectives)

            best, worst = replacements[-1]
            kept = 0
            _, g, f = 0, 1, 2
            
            x = [scrambled_objectives[c][-1][f] for c in [kept, worst, best]]
            y = [scrambled_objectives[c][-1][g] for c in [kept, worst, best]]
            
            pt.plot(x[:2], y[:2], 'ro-')
            pt.plot(x[1:], y[1:], 'ko-')

        pt.xlabel("folkliness")
        pt.ylabel("godliness")
        pt.title("Red kept, to worst to best")
        pt.show()

        reps = list(range(num_reps))
        for rep in reps:
            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            if not os.path.exists("%s/%s.pkl" % (dump_dir, dump_name)): break
            with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
                (rules, logs, choices, objectives, ema_objectives, replacements, batch_times) = pk.load(f)
            scrambled_objectives, uniform_objectives = objectives
            ema_objectives = np.stack(ema_objectives)

            for c in range(num_candidates):

                toggle_link_choices, macro_link_choices = choices[c]
                _, godliness, folkliness = scrambled_objectives[c][-1]

                ts, lents = zip(*toggle_link_choices)
                chain_portions = np.array(ts) / np.array(lents)

                pt.subplot(1,3,1)
                pt.scatter(ts, lents, color='k')
                pt.xlabel("choice")
                pt.ylabel("num choices")

                pt.subplot(1,3,2)
                pt.scatter(chain_portions, [godliness] * len(chain_portions), color='k')
                pt.xlabel("chain portion")
                pt.ylabel("godliness")

                pt.subplot(1,3,3)
                pt.scatter(chain_portions, [folkliness] * len(chain_portions), color='k')
                pt.xlabel("chain portion")
                pt.ylabel("folkliness")

        pt.show()



