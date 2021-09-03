import numpy as np
from relaxed_construction import Constructor, rewind
from utils import softmax
from pattern_database import PatternDatabase
from algorithm import run
from approximate_construction import scrambled, uniform
import matplotlib.pyplot as pt

def inc_batch(
    batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
    constructor, scrambled_objectives, uniform_objectives, num_problems,
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

def inc_on(
    batch, inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
    constructor, scrambled_objectives, uniform_objectives, num_problems,
):

    for k, (state, path) in enumerate(batch):

        if constructor.num_rules == constructor.max_rules: break
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
    do_cons = True
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

    num_problems = 32
    eval_period = 100
    correctness_bar = 1.1
    gamma = .99
    correctness_bar = 1.0

    inc_sampler = "scrambled"
    eval_samplers = ["scrambled", "uniform"]
    assert inc_sampler in eval_samplers

    num_candidates = 16
    max_iters = 10000
    ema_factor = 0
    keep_one = int(True)

    breakpoint = -1
    # breakpoint = 100
    num_reps = 30
    break_seconds = 0 * 60
    verbose = True

    # set up descriptive dump name
    dump_period = 10000
    dump_dir = "dtw"
    dump_base = "N%d%s_D%d_M%d_cn%d_%s_P%d_e%s" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral, inc_sampler, num_candidates, ema_factor)

    import itertools as it
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

            objective_archive = []
            scrambled_objectives = [[] for _ in range(num_candidates)]
            uniform_objectives = [[] for _ in range(num_candidates)]
            exhaust_objectives = [[] for _ in range(num_candidates)]
            ema_objectives = [[] for _ in range(num_candidates)]

            scalarize_archive = []
            scrambled_scalarized = [[] for _ in range(num_candidates)]
            uniform_scalarized = [[] for _ in range(num_candidates)]
            exhaust_scalarized = [[] for _ in range(num_candidates)]
            ema_scalarized = [[] for _ in range(num_candidates)]

            # # initial scalarization weights in S+
            weights = rng.normal(size=2) # random
            # weights = np.array([0.01, .99]) # folksy
            # weights = np.array([.99, .01]) # godly
            # weights = np.array([.5, .5]) # equally important
            weights = np.fabs(weights)
            weights /= np.sqrt(np.sum(weights**2))

            def scalarize(objs):
                return np.maximum(0, np.array(objs) / weights).min(axis=-1) ** 2

            replacements = []
            current_objectives = np.empty((num_candidates, 3))

            iter_times = []

            for k in range(max_iters):

                iter_start = perf_counter()

                batch_size = num_problems if k == 0 else 1
                if inc_sampler == "scrambled": probs = [scrambled_sample() for _ in range(batch_size)]
                if inc_sampler == "uniform": probs = [uniform_sample() for _ in range(batch_size)]

                for c in range(num_candidates):

                    correctness, godliness, folkliness = constructors[c].evaluate(probs)
                    if k > 0:
                        correctness = gamma * ema_objectives[c][-1][0] + (1 - gamma) * correctness
                        godliness = gamma * ema_objectives[c][-1][1] + (1 - gamma) * godliness
                    ema_objectives[c].append((correctness, godliness, folkliness))
                    ema_scalarized[c].append(scalarize((godliness, folkliness)))
                    current_objectives[c] = correctness, godliness, folkliness

                    constructors[c].incorporate(*probs[0])
                    # correctness, godliness, folkliness = inc_on(batch,
                    #     inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
                    #     constructors[c], scrambled_objectives[c], uniform_objectives[c], num_problems,
                    # )
                    # correctness, godliness, folkliness = inc_batch(
                    #     batch_size if k > 0 else initial_batch,
                    #     inc_sampler, eval_samplers, scrambled_sample, uniform_sample,
                    #     constructors[c], scrambled_objectives[c], uniform_objectives[c], num_problems,
                    # )

                    if k % eval_period == 0:
    
                        if "scrambled" in eval_samplers:
                            probs = [scrambled_sample() for _ in range(num_problems)]
                            scrambled_objectives[c].append(constructors[c].evaluate(probs))
                            scrambled_scalarized[c].append(scalarize(scrambled_objectives[c][-1][1:]))
                    
                        if "uniform" in eval_samplers:
                            probs = [uniform_sample() for _ in range(num_problems)]
                            uniform_objectives[c].append(constructors[c].evaluate(probs))
                            uniform_scalarized[c].append(scalarize(uniform_objectives[c][-1][1:]))
    
                        if inc_sampler == "scrambled": all_probs = [scrambled_sample() for _ in range(len(all_states))]
                        if inc_sampler == "uniform": all_probs = list(zip(all_states, optimal_paths))

                        exhaust_obj = constructors[c].evaluate(all_probs)
                        exhaust_objectives[c].append(exhaust_obj)
                        exhaust_scalarized[c].append(scalarize(exhaust_obj[1:]))

                # scalarize all candidates
                scalarized = scalarize(current_objectives[:,1:])

                # replace the worst with copy of the best
                worst = np.argmin(scalarized[keep_one:]) + keep_one
                best = np.argmax(scalarized[keep_one:]) + keep_one
                replacements.append((best, worst))

                if verbose:
                    print("%d, %d: %d [%d r, ~%f s, ~%f g, ~%f f] %f --> %d [%d r, ~%f s, ~%f g, ~%f f] %f cf [%d r, ~%f s, ~%f g, ~%f f] %f (%f)" % (
                        rep, k,
                        worst, constructors[worst].num_rules, current_objectives[worst,0], current_objectives[worst,1], current_objectives[worst,2], scalarized[worst],
                        best, constructors[best].num_rules, current_objectives[best,0], current_objectives[best,1], current_objectives[best,2], scalarized[best],
                        constructors[0].num_rules, current_objectives[0,0], current_objectives[0,1], current_objectives[0,2], scalarized[0],
                        exhaust_objectives[best][-1][0]
                    ))

                objectives = (scrambled_objectives, uniform_objectives, exhaust_objectives, ema_objectives)
                scalarized = (scrambled_scalarized, uniform_scalarized, exhaust_scalarized, ema_scalarized)
                objective_archive.append(tuple(objs[worst] for objs in objectives))
                scalarize_archive.append(tuple(scas[worst] for scas in scalarized))
                for objs in objectives: objs[worst] = list(objs[best])
                for scas in scalarized: scas[worst] = list(scas[best])
                constructors[worst] = constructors[best].copy()
                current_objectives[worst] = current_objectives[best]

                iter_times.append(perf_counter() - iter_start)

                if k % dump_period == 0:
                    dump_name = "%s_r%d" % (dump_base, rep)
                    rules = [constructors[c].rules() for c in range(num_candidates)]
                    logs = [constructors[c].logs() for c in range(num_candidates)]
                    choices = [constructors[c].choices() for c in range(num_candidates)]
                    with open(dump_name + ".pkl", "wb") as f:
                        pk.dump((rules, logs, choices, weights,
                            scalarized, scalarize_archive,
                            objectives, objective_archive,
                            replacements, iter_times), f)

                # if correctness >= correctness_bar: break
                if exhaust_objectives[best][-1][0] == 1.0: break

            rep_time = perf_counter() - rep_start

            dump_name = "%s_r%d" % (dump_base, rep)
            print(dump_name)
            objectives = (scrambled_objectives, uniform_objectives, exhaust_objectives, ema_objectives)
            scalarized = (scrambled_scalarized, uniform_scalarized, exhaust_scalarized, ema_scalarized)
            rules = [constructors[c].rules() for c in range(num_candidates)]
            logs = [constructors[c].logs() for c in range(num_candidates)]
            choices = [constructors[c].choices() for c in range(num_candidates)]
            with open(dump_name + ".pkl", "wb") as f:
                pk.dump((rules, logs, choices, weights,
                    scalarized, scalarize_archive,
                    objectives, objective_archive,
                    replacements, iter_times), f)
            os.system("mv %s.pkl %s/%s.pkl" % (dump_name, dump_dir, dump_name))

            if verbose:
                print("Took %s seconds total." % rep_time)
                print("Breaking for %s seconds..." % str(break_seconds))

            sleep(break_seconds)

    if show_results:

        # rep = 0
        # dump_name = "%s_r%d" % (dump_base, rep)
        # with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
        #     (rules, logs, choices, weights,
        #     scalarized, scalarize_archive,
        #     objectives, objective_archive,
        #     replacements, iter_times) = pk.load(f)
        # scrambled_objectives, uniform_objectives, exhaust_objectives, ema_objectives = objectives
        # scrambled_scalarized, uniform_scalarized, exhaust_scalarized, ema_scalarized = scalarized
        # pt.subplot(1,4,1)
        # for (scrambled, uniform, exhaust, ema) in scalarize_archive:
        #     # pt.plot(exhaust, '-', color=(.75,)*3)
        #     pt.plot(ema, '-', color=(.75,)*3)
        # for c in range(num_candidates):
        #     color = (0,)*3 if c > 0 else 'b'
        #     # pt.plot(exhaust_scalarized[c], '-', color=color)
        #     pt.plot(ema_scalarized[c], '-', color=color)
        # pt.xlabel("batch")
        # pt.ylabel("Scalarized")
        # for n,name in enumerate(("Correctness", "Godliness", "Folksiness")):
        #     pt.subplot(1,4,n+2)
        #     for (scrambled, uniform, exhaust, ema) in objective_archive:
        #         # pt.plot([obj[n] for obj in exhaust], '-', color=(.75,)*3)
        #         pt.plot([obj[n] for obj in ema], '-', color=(.75,)*3)
        #     for c in range(num_candidates):
        #         color = (0,)*3 if c > 0 else 'b'
        #         # pt.plot([obj[n] for obj in exhaust_objectives[c]], '-', color=color)
        #         pt.plot([obj[n] for obj in ema_objectives[c]], '-', color=color)
        #     pt.xlabel("batch")
        #     pt.ylabel(name)
        # pt.show()

        for rep in range(num_reps):
            dump_name = "%s_r%d" % (dump_base, rep)
            dump_path = "%s/%s.pkl" % (dump_dir, dump_name)
            if not os.path.exists(dump_path): continue

            print(dump_name)
            with open(dump_path, "rb") as f:
                (rules, logs, choices, weights,
                scalarized, scalarize_archive,
                objectives, objective_archive,
                replacements, iter_times) = pk.load(f)
            scrambled_objectives, uniform_objectives, exhaust_objectives, ema_objectives = objectives
            scrambled_scalarized, uniform_scalarized, exhaust_scalarized, ema_scalarized = scalarized

            pt.subplot(1,2,1)
            scalarized = np.array([ema_scalarized[c][-1] for c in range(num_candidates)])
            best = scalarized.argmax()
            # pt.plot(ema_objectives[0][-1][2], ema_objectives[0][-1][1], 'b.')
            # pt.plot(ema_objectives[best][-1][2], ema_objectives[best][-1][1], 'k.')
            pt.plot(
                [ema_objectives[0][-1][2], ema_objectives[best][-1][2]],
                [ema_objectives[0][-1][1], ema_objectives[best][-1][1]],
                'bo-')
            pt.plot(ema_objectives[best][-1][2], ema_objectives[best][-1][1], 'ko')
            pt.xlabel("Folkliness")
            pt.ylabel("Godliness")
            pt.title("EMA")
            pt.subplot(1,2,2)
            scalarized = np.array([exhaust_scalarized[c][-1] for c in range(num_candidates)])
            best = scalarized.argmax()
            # pt.plot(exhaust_objectives[0][-1][2], exhaust_objectives[0][-1][1], 'b.')
            # pt.plot(exhaust_objectives[best][-1][2], exhaust_objectives[best][-1][1], 'k.')
            pt.plot(
                [exhaust_objectives[0][-1][2], exhaust_objectives[best][-1][2]],
                [exhaust_objectives[0][-1][1], exhaust_objectives[best][-1][1]],
                'bo-')
            pt.plot(exhaust_objectives[best][-1][2], exhaust_objectives[best][-1][1], 'ko')
            pt.xlabel("Folkliness")
            pt.ylabel("Godliness")
            pt.title("True")
        
        pt.show()

        # reps = list(range(3))
        # # reps = list(range(num_reps))
        # for rep in reps:
        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     print(dump_name)
        #     with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
        #         (rules, logs, choices, objectives, ema_objectives, replacements, iter_times) = pk.load(f)

        #     # discontinuities in _objectives when a worst is replaced; don't plot fully connected
        #     scrambled_objectives, uniform_objectives = objectives
        #     ema_objectives = np.stack(ema_objectives)
    
        #     for c in range(num_candidates):
        #         print(c)
        #         patterns, wildcards, macros = rules[c]
        #         num_rules, num_incs, inc_added, inc_disabled, chain_lengths = logs[c]
                
        #         # correctness, godliness, folkliness = zip(*scrambled_objectives[c])
        #         correctness, godliness, folkliness = ema_objectives[:,:,0], ema_objectives[:,:,1], ema_objectives[:,:,2]
        #         folkliness = np.array([(inc_added < i).sum() for i in range(0,num_incs, batch_size)])
                
        #         print("these are wrong, discontinuous objective lists")
        #         pt.subplot(len(reps), 3, rep*3 + 1)
        #         pt.plot(correctness)
        #         pt.xlabel("batches")
        #         pt.ylabel("correctness")
        #         pt.subplot(len(reps), 3, rep*3 + 2)
        #         pt.plot(godliness)
        #         pt.xlabel("batches")
        #         pt.ylabel("godliness")
        #         pt.subplot(len(reps), 3, rep*3 + 3)
        #         pt.plot(folkliness)
        #         pt.xlabel("batches")
        #         pt.ylabel("folkliness")
        #         pt.title("Should have disconnections")

        # pt.show()

        # reps = list(range(num_reps))
        # for rep in reps:
        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     print(dump_name)
        #     if not os.path.exists("%s/%s.pkl" % (dump_dir, dump_name)): break
        #     with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
        #         (rules, logs, choices, objectives, ema_objectives, replacements, iter_times) = pk.load(f)
        #     scrambled_objectives, uniform_objectives = objectives
        #     ema_objectives = np.stack(ema_objectives)

        #     best, worst = replacements[-1]
        #     kept = 0
        #     _, g, f = 0, 1, 2
            
        #     x = [scrambled_objectives[c][-1][f] for c in [kept, worst, best]]
        #     y = [scrambled_objectives[c][-1][g] for c in [kept, worst, best]]
            
        #     pt.plot(x[:2], y[:2], 'ro-')
        #     pt.plot(x[1:], y[1:], 'ko-')

        # pt.xlabel("folkliness")
        # pt.ylabel("godliness")
        # pt.title("Red kept, to worst to best")
        # pt.show()

        # reps = list(range(num_reps))
        # for rep in reps:
        #     dump_name = "%s_r%d" % (dump_base, rep)
        #     print(dump_name)
        #     if not os.path.exists("%s/%s.pkl" % (dump_dir, dump_name)): break
        #     with open("%s/%s.pkl" % (dump_dir, dump_name), "rb") as f:
        #         (rules, logs, choices, objectives, ema_objectives, replacements, iter_times) = pk.load(f)
        #     scrambled_objectives, uniform_objectives = objectives
        #     ema_objectives = np.stack(ema_objectives)

        #     for c in range(num_candidates):

        #         toggle_link_choices, macro_link_choices = choices[c]
        #         _, godliness, folkliness = scrambled_objectives[c][-1]

        #         ts, lents = zip(*toggle_link_choices)
        #         chain_portions = np.array(ts) / np.array(lents)

        #         pt.subplot(1,3,1)
        #         pt.scatter(ts, lents, color='k')
        #         pt.xlabel("choice")
        #         pt.ylabel("num choices")

        #         pt.subplot(1,3,2)
        #         pt.scatter(chain_portions, [godliness] * len(chain_portions), color='k')
        #         pt.xlabel("chain portion")
        #         pt.ylabel("godliness")

        #         pt.subplot(1,3,3)
        #         pt.scatter(chain_portions, [folkliness] * len(chain_portions), color='k')
        #         pt.xlabel("chain portion")
        #         pt.ylabel("folkliness")

        # pt.show()



