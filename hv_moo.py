import numpy as np
import matplotlib.pyplot as pt
import pickle as pk
import os
from time import sleep, perf_counter
from relaxed_construction import Constructor
from approximate_construction import scrambled, uniform

class Trail:
    def __init__(self, constructor, batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample, sample_size):
    
        self.constructor = constructor
        self.batch_size = batch_size
        self.inc_sampler = inc_sampler
        self.eval_samplers = eval_samplers
        self.scrambled_sample = scrambled_sample
        self.uniform_sample = uniform_sample
        self.sample_size = sample_size

        self.scrambled_objectives = []
        self.uniform_objectives = []

    def copy(self):

        trail = Trail(self.constructor.copy(), batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample, sample_size)
        trail.scrambled_objectives = list(self.scrambled_objectives)
        trail.uniform_objectives = list(self.uniform_objectives)
        return trail

    def objectives(self):
        return self.scrambled_objectives, self.uniform_objectives

    def inc_batch(self):

        for k in range(self.batch_size):
            if self.constructor.num_rules == self.constructor.max_rules: break
            if self.inc_sampler == "scrambled": state, path = self.scrambled_sample()
            if self.inc_sampler == "uniform": state, path = self.uniform_sample()    
            self.constructor.incorporate(state, path)
    
        if "scrambled" in self.eval_samplers:
            probs = [scrambled_sample() for _ in range(self.sample_size)]
            self.scrambled_objectives.append(self.constructor.evaluate(probs))
    
        if "uniform" in eval_samplers:
            probs = [uniform_sample() for _ in range(self.sample_size)]
            self.uniform_objectives.append(self.constructor.evaluate(probs))
    
        if inc_sampler == "scrambled": correctness, godliness, folkliness = self.scrambled_objectives[-1]
        if inc_sampler == "uniform": correctness, godliness, folkliness = self.uniform_objectives[-1]
        
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

    inc_sampler = "scrambled"
    # eval_samplers = ["scrambled", "uniform"]
    eval_samplers = ["scrambled"]
    assert inc_sampler in eval_samplers

    beam_size = 8
    sample_size = 32
    batch_size = 32
    correctness_bar = 1.0
    ema_factor = .9
    keep_one = int(False)

    break_seconds = 2 * 60
    break_period = 1000

    num_reps = 1
    num_scalarizations = 1
    max_constructors_per_rep = 16 + beam_size
    max_rules = None

    verbose = True
    do_cons = False
    show_results = True
    confirm = False
    confirm_show = False

    # set up descriptive dump name
    dump_period = 1000
    dump_dir = "hvm"
    dump_base = "N%da%dq%d_D%d_M%d_cn%d_%s_P%d_B%d_e%s_S%d" % (
        cube_size, num_twist_axes, quarter_turns,
        tree_depth, max_depth, color_neutral, inc_sampler,
        beam_size, batch_size, ema_factor,
        num_scalarizations,
    )

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
    if max_rules is None: max_rules = len(all_states)

    rng = np.random.default_rng()
    uniform_sample = uniform(rng, all_states, optimal_paths)
    scrambled_sample = scrambled(domain, rng, max_scramble_length)

    if do_cons:

        max_godly = 0
        max_folkly = 0

        # initialize moos
        terminated = np.zeros((num_reps, num_scalarizations), dtype=bool)
        current_correctness = np.empty((num_reps, num_scalarizations, beam_size))
        current_num_rules = np.empty((num_reps, num_scalarizations, beam_size), dtype=int)
        current_objectives = np.empty((num_reps, num_scalarizations, beam_size, 2))
        trails = {}
        for r,s,c in it.product(range(num_reps), range(num_scalarizations), range(beam_size)):
            constructor = Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
            trails[r,s,c] = Trail(constructor, batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample, sample_size)
            current_correctness[r,s,c], godliness, folkliness = trails[r,s,c].inc_batch()
            current_num_rules[r,s,c] = trails[r,s,c].constructor.num_rules
            current_objectives[r,s,c, 0] = godliness
            current_objectives[r,s,c, 1] = folkliness

        # initial scalarization weights in S+
        weights = rng.normal(size=(num_reps, num_scalarizations, 2))
        weights /= np.sqrt(np.sum(weights**2, axis=2))[:,:,np.newaxis]
        weights = np.fabs(weights)

        objective_archive = [[] for r in range(num_reps)]
        frontiers = [[] for r in range(num_reps)]

        # run moos in parallel
        for parallel_iters in it.count():
            if terminated.all(): break

            # incorporate new states into all trails
            for r,s,c in it.product(range(num_reps), range(num_scalarizations), range(beam_size)):
                if terminated[r,s]: continue
                current_correctness[r,s,c], godliness, folkliness = trails[r,s,c].inc_batch()
                current_num_rules[r,s,c] = trails[r,s,c].constructor.num_rules
                current_objectives[r,s,c,0] = (1 - ema_factor) * godliness + ema_factor * current_objectives[r,s,c,0]
                current_objectives[r,s,c,1] = folkliness

            # update beams
            for r,s in it.product(range(num_reps), range(num_scalarizations)):
                if terminated[r,s]: continue

                # compute scalarizations
                # scalarized = np.maximum(0, current_objectives[r,s] / weights[r,s]).min(axis=1)**2 # constructors are axis=0
                scalarized = current_objectives[r,s,:,0] # godliness
                # scalarized = current_objectives[r,s,:,1] # folkliness

                # check best and worst for next beam
                worst = np.argmin(scalarized[keep_one:]) + keep_one
                best = np.argmax(scalarized[keep_one:]) + keep_one

                # check for converged beam search
                if current_correctness[r,s,best] >= correctness_bar or trails[r,s,best].constructor.num_rules == max_rules:

                    # archive frontier solution
                    frontiers[r].append((
                        trails[r,s,best].constructor.rules(),
                        trails[r,s,best].objectives(),
                        tuple(weights[r,s])))
                    for c in range(beam_size):
                        scrambled_objectives, uniform_objectives = trails[r,s,c].objectives()
                        objective_archive[r].append((scrambled_objectives, uniform_objectives))
                        if c == best:
                            _, godliness, folkliness = scrambled_objectives[-1]
                            max_godly = max(max_godly, godliness)
                            max_folkly = max(max_folkly, folkliness)

                    # terminate beam search if constructors are maxed out
                    current_constructors = len(frontiers[r]) + num_scalarizations * beam_size
                    if current_constructors >= max_constructors_per_rep:
                        terminated[r,s] = True
                    else:

                        # initialize new scalarization beam
                        for c in range(beam_size):
                            constructor = Constructor(max_rules, rng, domain, tree, max_depth, max_actions, use_safe_depth, color_neutral)
                            trails[r,s,c] = Trail(constructor, batch_size, inc_sampler, eval_samplers, scrambled_sample, uniform_sample, sample_size)
                            current_correctness[r,s,c], godliness, folkliness = trails[r,s,c].inc_batch()
                            current_num_rules[r,s,c] = trails[r,s,c].constructor.num_rules
                            current_objectives[r,s,c] = (godliness, folkliness)
                
                        # initialize new scalarization weights in S+
                        weights[r,s] = rng.normal(size=2)
                        weights[r,s] /= np.sqrt(np.sum(weights[r,s]**2))
                        weights[r,s] = np.fabs(weights[r,s])

                else:

                    # not converged, replace worst with best
                    objective_archive[r].append(trails[r,s,worst].objectives())
                    trails[r,s,worst] = trails[r,s,best].copy()
                    current_correctness[r,s,worst] = current_correctness[r,s,best]
                    current_num_rules[r,s,worst] = current_num_rules[r,s,best]
                    current_objectives[r,s,worst] = current_objectives[r,s,best]

            if verbose:
                print("%d: %s terminated. %d-%d<=%d pioneers, maxes g|f: %f | %f. current r|c|g|f: %d-%d | %f-%f | %f-%f | %f-%f" % (
                    parallel_iters, str(terminated.sum() / terminated.size),
                    min(map(len, frontiers)), max(map(len, frontiers)), max_constructors_per_rep,
                    max_godly, max_folkly,
                    current_num_rules.min(), current_num_rules.max(),
                    current_correctness.min(), current_correctness.max(),
                    current_objectives[:,:,:,0].min(), current_objectives[:,:,:,0].max(),
                    current_objectives[:,:,:,1].min(), current_objectives[:,:,:,1].max(),
                ))

            if parallel_iters % dump_period == 0:
                with open(dump_base + ".pkl", "wb") as f:
                    pk.dump((frontiers, objective_archive), f)

            if parallel_iters % break_period == break_period - 1:
                if verbose: print("Breaking for %s seconds..." % str(break_seconds))
                sleep(break_seconds)

        os.system("mv %s.pkl %s/%s.pkl" % (dump_base, dump_dir, dump_base))

    if show_results:
        
        print("loading %s" % dump_base)
        with open("%s/%s.pkl" % (dump_dir, dump_base), "rb") as f:
            frontiers, objective_archive = pk.load(f)
        print("loaded")

        for r in range(num_reps):
            pt.subplot(1, num_reps, r+1)
            
            idx = np.random.choice(len(objective_archive[r]), size=2000)
            # for b,(scrambled_objectives, uniform_objectives) in enumerate(objective_archive[r][-2000:]):
                # if b == 2000: break
            for b in idx:
                scrambled_objectives, uniform_objectives = objective_archive[r][b]
                print("%d,%d of %d" % (r,b, len(objective_archive[r])))
                # correctness, godliness, folkliness = zip(*scrambled_objectives)
                # pt.plot(folkliness, godliness, '-', color=(.5, .5, .5))
                # pt.scatter(folkliness[-1], godliness[-1], color=(.5,)*3)
                _, godliness, folkliness = zip(*scrambled_objectives)
                godliness = list(godliness)
                for g in range(1,len(godliness)):
                    godliness[g] = .99 * godliness[g-1] + (1 - .99) * godliness[g]
                pt.plot(folkliness[-1], godliness[-1], '.', color=(.5,)*3)
                # _, godliness, folkliness = scrambled_objectives[-1]
                # pt.plot(folkliness, godliness, '.', color=(.5,)*3)

            if len(frontiers[r]) > 0:
                rules, objectives, weights = zip(*frontiers[r])
                for f,(scrambled_objectives, uniform_objectives) in enumerate(objectives):
                    print("%d,f %d of %d" % (r,f, len(objectives)))
                    correctness, godliness, folkliness = zip(*scrambled_objectives)
                    godliness = list(godliness)
                    for g in range(1,len(godliness)):
                        godliness[g] = .99 * godliness[g-1] + (1 - .99) * godliness[g]
                    pt.plot(folkliness[100:], godliness[100:], '-k')
                    pt.plot(folkliness[-1], godliness[-1], 'ro')
                    # pt.scatter(folkliness[-1], godliness[-1], color=(0.,)*3)
                    # _, godliness, folkliness = scrambled_objectives[-1]
                    # pt.plot(folkliness, godliness, '.', color=(0.,)*3)
            
        pt.show()
