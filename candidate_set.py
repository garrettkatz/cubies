import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from pattern_database import PatternDatabase
from algorithm import run
from utils import softmax

class Candidate:
    def __init__(self, patterns, wildcard, macros):
        self.patterns = patterns
        self.wildcard = wildcard
        self.macros = macros
        self.scramble_counts = None
        self.match_counts = None
        self.miss_counts = None
        self.num_queries = None
        self.macro_counts = None
        self.action_counts = None
        self.successes = None

class CandidateSet:

    def __init__(self, domain, bfs_tree, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length, num_patterns, num_instances, max_depth, max_macros):
        self.domain = domain
        self.bfs_tree = bfs_tree
        self.rng = rng
        self.min_macro_size = min_macro_size
        self.max_macro_size = max_macro_size
        self.wildcard_rate = wildcard_rate
        self.rollout_length = rollout_length
        self.num_patterns = num_patterns
        self.num_instances = num_instances
        self.max_depth = max_depth
        self.max_macros = max_macros

    def sample_macro(self):
        macro_size = self.rng.integers(self.min_macro_size, self.max_macro_size, endpoint=True)
        lo = self.rng.choice(self.rollout_length - macro_size)
        hi = lo + macro_size
        actions = self.rng.choice(list(self.domain.valid_actions()), size=self.rollout_length, replace=True)
        macro = self.domain.reverse(actions[lo:hi])
        pattern = self.domain.execute(actions[:hi], self.domain.solved_state())
        return pattern, macro

    def spawn(self):

        # sample macros with valid patterns
        patterns = []
        macros = []
        for p in range(self.num_patterns):
            pattern, macro = self.sample_macro()
            patterns.append(pattern)
            macros.append(macro)

        # sample wildcards
        wildcard = (self.rng.random((len(patterns), self.domain.state_size())) < self.wildcard_rate)

        return Candidate(patterns, wildcard, macros)

    def evaluate(self, candidate):

        # initialize traces
        candidate.scramble_counts = np.empty(self.num_instances, dtype=int)
        candidate.match_counts = np.empty((self.num_instances, len(candidate.patterns)), dtype=int)
        candidate.miss_counts = np.zeros((len(candidate.patterns), len(candidate.patterns[0])), dtype=int)
        candidate.num_queries = np.empty(self.num_instances, dtype=int)
        candidate.macro_counts = np.empty(self.num_instances, dtype=int)
        candidate.action_counts = np.empty(self.num_instances, dtype=int)
        candidate.successes = np.empty(self.num_instances, dtype=bool)

        ### Run candidate on problem instances
        pattern_database = PatternDatabase(candidate.patterns, candidate.wildcard, candidate.macros, self.domain)

        # before sym aggregate
        miss_counts = np.zeros(pattern_database.miss_counts.shape, dtype=int)

        for i in range(self.num_instances):

            # Run algorithm on instance
            candidate.scramble_counts[i] = self.rng.integers(1, self.rollout_length, endpoint=True)
            state = self.domain.random_state(candidate.scramble_counts[i], self.rng)
            solved, plan = run(state, self.domain, self.bfs_tree, pattern_database, self.max_depth, self.max_macros)
            candidate.successes[i] = solved

            # Record plan length and macro count
            candidate.action_counts[i] = 0
            for (actions, sym, macro) in plan: candidate.action_counts[i] += len(actions) + len(macro)
            candidate.macro_counts[i] = len(plan)

            # Update database metrics
            candidate.match_counts[i] = pattern_database.match_counts
            candidate.num_queries[i] = pattern_database.num_queries
            miss_counts += pattern_database.miss_counts

            pattern_database.reset()

        # aggregate miss counts over symmetries
        perms = self.domain.symmetries_of(np.arange(self.domain.state_size()))
        for p in range(len(candidate.patterns)):
            for s in range(24):
                perm = perms[pattern_database.syms[24*p + s]]
                candidate.miss_counts[p] += pattern_database.miss_counts[24*p + s][perm]
    
        ### Evaluate objective functions
        # solve in <= 20 steps
        # solve in <= scramble steps
        # smaller number of patterns
        # smaller macro lengths
        # less complex patterns (like colors grouped together, more wildcards)?
        
        pattern_size = -len(candidate.patterns)
        macro_size = -sum(map(len, candidate.macros))
        godly_solves = (candidate.successes & (
            candidate.action_counts <= np.minimum(self.domain.god_number(), candidate.scramble_counts)
        )).sum()
    
        objectives = (pattern_size, macro_size, godly_solves)

        return candidate, objectives

    def mutate_scores(self, candidate):
        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]

        # change one pattern+macro based on match counts
        # scores = candidate.good_match_counts - candidate.fail_match_counts
        scores = candidate.match_counts[candidate.successes].sum(axis=0) - candidate.match_counts[~candidate.successes].sum(axis=0)
        # p = np.argmin(scores) # too hard, can inhibit exploration
        p = self.rng.choice(len(candidate.patterns), p= softmax(-scores))
        patterns[p], macros[p] = self.sample_macro()

        return Candidate(patterns, wildcard, macros)

    def mutate_unguided(self, candidate):

        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]
        wildcard = candidate.wildcard.copy()

        # mutation types:
        # change one (or none) wildcard of one pattern
        p = self.rng.choice(len(patterns))
        i = self.rng.choice(len(patterns[p]))
        wildcard[p, i] = self.rng.choice([True, False])

        # perform one (or none) twist to one pattern state
        p = self.rng.choice(len(patterns))
        axis, plane, twist = self.rng.choice(list(it.product([0,1,2], range(self.domain.N), [-1,0,1])))
        patterns[p] = self.domain.perform((axis, plane, twist), patterns[p])

        # change one (or none) action of one macro
        m = self.rng.choice(len(macros))
        a = self.rng.integers(self.max_macro_size)
        if a < len(macros[m]): macros[m][a] = tuple(self.rng.choice(self.domain.valid_actions()))

        # delete one (or none) action of one macro
        m = self.rng.choice(len(macros))
        if len(macros[m]) > self.min_macro_size:
            a = self.rng.integers(self.max_macro_size)
            if a < len(macros[m]): macros[m] = macros[m][:a] + macros[m][a+1:]

        # add one (or none) action to one macro
        m = self.rng.choice(len(macros))
        if len(macros[m]) < self.max_macro_size:
            macros[m] = macros[m] + [tuple(self.rng.choice(self.domain.valid_actions()))]

        return Candidate(patterns, wildcard, macros)

    def mutate_macro(self, candidate):

        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]

        # mutation types:
        # change one element of one pattern to wildcard
        p = self.rng.choice(len(patterns))
        patterns[p][self.rng.choice(len(patterns[p]))] = 0

        # add, delete, or change one action of one macro
        # choosing the macro: prefer ones that were used a lot and usually failed
        # match_counts: num instance x num patterns|macros
        probs = softmax(np.sqrt(candidate.match_counts.sum(axis=0) * candidate.match_counts[candidate.successes].sum(axis=0)))

        # choosing add, delete or change depends on two things:
        # if usually fails early, add (correlated fails and earlies)
        # if usually fails at end, del (correlated fails and lates)
        # if fails but evenly at different steps, change (fails not correlated with endpoints)
        # or, do each

        # change one action of one macro
        m = self.rng.choice(len(macros), p = probs)
        a = self.rng.integers(self.max_macro_size, endpoint=True)
        if a < len(macros[m]): macros[m][a] = tuple(self.rng.choice(self.domain.valid_actions()))

        # delete one action of one macro
        m = self.rng.choice(len(macros), p = probs)
        if len(macros[m]) > self.min_macro_size:
            a = self.rng.integers(self.max_macro_size, endpoint=True)
            if a < len(macros[m]): macros[m] = macros[m][:a] + macros[m][a+1:]

        # add one action to one macro
        m = self.rng.choice(len(macros), p = probs)
        if len(macros[m]) < self.max_macro_size:
            macros[m] = macros[m] + [tuple(self.rng.choice(self.domain.valid_actions()))]

        # change one pattern+macro
        p = self.rng.choice(len(patterns), p = probs)
        patterns[p], macros[p] = self.sample_macro()

        return Candidate(patterns, wildcard, macros)

    def mutate_guided(self, candidate):

        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]
        wildcard = candidate.wildcard.copy()

        # if unmatched, make more wildcards
        match_counts = candidate.match_counts.sum(axis=0)
        m = self.rng.choice(len(macros), p = softmax(-match_counts))
        # m = (-match_counts).argmax()
        # if not wildcard[m].all():
        #     w = self.rng.choice(np.flatnonzero(~wildcard[m]))
        #     wildcard[m,w] = True
        w = self.rng.choice(wildcard.shape[1], p = softmax(candidate.miss_counts[m]))
        wildcard[m,w] = True

        # if matched and failing, sample new macro
        if not candidate.successes.all():
            # solve_scores = \
            #     candidate.match_counts[candidate.successes].sum(axis=0) - \
            #     candidate.match_counts[~candidate.successes].sum(axis=0)
            # m = self.rng.choice(len(macros), p = softmax(-solve_scores))
            fail_counts = candidate.match_counts[~candidate.successes].sum(axis=0)
            m = self.rng.choice(len(macros), p = softmax(fail_counts))
            # m = fail_counts.argmax()

            patterns[m], macros[m] = self.sample_macro()
            wildcard[m] = (self.rng.random(len(patterns[m])) < self.wildcard_rate)

        return Candidate(patterns, wildcard, macros)

    def show(self, candidate, cap=8):
        _, axs = pt.subplots(min(cap, len(candidate.patterns)), self.max_macro_size + 1)
        for p in range(min(cap, len(candidate.patterns))):
            state = candidate.patterns[p] * (1 - candidate.wildcard[p])
            self.domain.render(state, axs[p,0], x0=0, y0=0)
            for m in range(len(candidate.macros[p])):
                action = candidate.macros[p][m]
                state = self.domain.perform(action, state)
                self.domain.render(state, axs[p,m+1], x0=0, y0=0)
                axs[p,m+1].set_title(str(action))
            for m in range(self.max_macros+1):
                axs[p,m].axis("equal")
                axs[p,m].axis('off')
        axs[0,0].set_title("Patterns")
        # pt.tight_layout()
        pt.show()
    
if __name__ == "__main__":
    
    cube_size = 2
    num_instances = 512
    tree_depth = 3
    max_depth = 1
    max_macros = 5
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 5
    wildcard_rate = .1
    rollout_length = 20

    from cube import CubeDomain
    domain = CubeDomain(cube_size)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, tree_depth)

    import numpy as np
    rng = np.random.default_rng()

    candidate_set = CandidateSet(
        domain, bfs_tree, rng, min_macro_size, max_macro_size, wildcard_rate, rollout_length,
        num_patterns, num_instances, max_depth, max_macros)

    candidate, objectives = candidate_set.evaluate(candidate_set.spawn())
    pattern_size, macro_size, godly_solves = objectives
    print()
    print("num_patterns = %d" % num_patterns)
    print("macro_size = %d" % macro_size)
    print("godly_solves = %d" % godly_solves)

    candidate, objectives = candidate_set.evaluate(candidate_set.mutate_unguided(candidate))
    pattern_size, macro_size, godly_solves = objectives

    print()
    print("num_patterns = %d" % num_patterns)
    print("macro_size = %d" % macro_size)
    print("godly_solves = %d" % godly_solves)

    # test mutate_guided
    lc = []
    mc = []
    wc = []
    candidate, objectives = candidate_set.evaluate(candidate_set.spawn())
    for n in range(1000):
        new_candidate, new_objectives = candidate_set.evaluate(candidate_set.mutate_guided(candidate))
        print(n, objectives[2], new_objectives[2])
        if new_objectives[2] > objectives[2]:
            candidate = new_candidate
            objectives = new_objectives
        lc.append(new_objectives[2])
        mc.append(new_candidate.match_counts.sum())
        wc.append(new_candidate.wildcard.sum())
    pt.subplot(1,3,1)
    pt.plot(lc)
    pt.plot([max(lc[:n+1]) for n in range(len(lc))])
    pt.ylabel("solves")
    pt.subplot(1,3,2)
    pt.plot(mc)
    pt.ylabel("match counts")
    pt.subplot(1,3,3)
    pt.plot(wc)
    pt.ylabel("wildcads")
    pt.show()

