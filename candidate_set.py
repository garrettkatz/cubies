import numpy as np
import matplotlib.pyplot as pt
from pattern_database import PatternDatabase
from algorithm import run
from utils import softmax

class Candidate:
    def __init__(self, patterns, macros):
        self.patterns = patterns
        self.macros = macros
        self.scramble_counts = None
        self.match_counts = None
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
        state = self.domain.execute(actions[:hi], self.domain.solved_state())
        pattern = state * (self.rng.random(state.shape) > self.wildcard_rate).astype(int)
        return pattern, macro
    
    def spawn(self):
        patterns = []
        macros = []
        for p in range(self.num_patterns):
            pattern, macro = self.sample_macro()
            patterns.append(pattern)
            macros.append(macro)
        return Candidate(patterns, macros)

    def evaluate(self, candidate):

        # initialize traces
        candidate.scramble_counts = np.empty(self.num_instances, dtype=int)
        candidate.match_counts = np.empty((self.num_instances, len(candidate.patterns)), dtype=int)
        candidate.num_queries = np.empty(self.num_instances, dtype=int)
        candidate.macro_counts = np.empty(self.num_instances, dtype=int)
        candidate.action_counts = np.empty(self.num_instances, dtype=int)
        candidate.successes = np.empty(self.num_instances, dtype=bool)

        ### Run candidate on problem instances
        pattern_database = PatternDatabase(candidate.patterns, candidate.macros, self.domain)

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
            pattern_database.reset()
    
        ### Evaluate objective functions
        # solve in <= 20 steps
        # solve in <= scramble steps
        # smaller number of patterns
        # smaller macro lengths
        # less complex patterns (like colors grouped together, more wildcards)?
        
        pattern_size = -len(candidate.patterns)
        macro_size = -sum(map(len, candidate.macros))
        godly_solves = (candidate.successes & (candidate.action_counts <= np.minimum(self.domain.god_number(), candidate.scramble_counts[i]))).sum()
    
        objectives = (pattern_size, macro_size, godly_solves)

        return candidate, objectives

    def mutate_scores(self, candidate):
        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]

        # change one pattern+macro based on match counts
        # scores = candidate.good_match_counts - candidate.fail_match_counts
        scores = candidate.match_counts[candidate.successes].sum(axis=0) - candidate.match_counts[~candidate.successes].sum(axis=0)
        # p = np.argmin(scores) # too hard, can inhibit exploration
        costs = -scores
        probs = np.exp(costs - costs.max())
        probs /= probs.sum()
        p = self.rng.choice(len(candidate.patterns), p=probs)
        patterns[p], macros[p] = self.sample_macro()

        return Candidate(patterns, macros)

    def mutate(self, candidate):

        patterns = [pattern.copy() for pattern in candidate.patterns]
        macros = [macro.copy() for macro in candidate.macros]

        # mutation types:
        # change one element of one pattern to wildcard
        p = self.rng.choice(len(patterns))
        patterns[p][self.rng.choice(len(patterns[p]))] = 0

        # change one action of one macro
        m = self.rng.choice(len(macros))
        a = self.rng.choice(len(macros[m]))
        macros[m][a] = tuple(self.rng.choice(self.domain.valid_actions()))

        # add or delete one action of one macro
        m = self.rng.choice(len(macros))
        a = self.rng.integers(self.max_macro_size, endpoint=True)
        if a < len(macros[m]) and self.min_macro_size < len(macros[m]):
            macros[m] = macros[m][:a] + macros[m][a+1:]
        else:
            macros[m] = macros[m] + [tuple(self.rng.choice(self.domain.valid_actions()))]

        # change one pattern+macro
        p = self.rng.choice(len(patterns))
        patterns[p], macros[p] = self.sample_macro()
        
        return Candidate(patterns, macros)

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

        return Candidate(patterns, macros)

    def show(self, candidate, cap=8):
        _, axs = pt.subplots(min(cap, len(candidate.patterns)), self.max_macro_size + 1)
        for p in range(min(cap, len(candidate.patterns))):
            state = candidate.patterns[p]
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
    max_scrambles = 5
    num_instances = 128
    tree_depth = 3
    max_depth = 1
    max_macros = 5
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 5
    wildcard_rate = .5
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

    candidate, objectives = candidate_set.evaluate(candidate_set.mutate(candidate))
    pattern_size, macro_size, godly_solves = objectives

    print()
    print("num_patterns = %d" % num_patterns)
    print("macro_size = %d" % macro_size)
    print("godly_solves = %d" % godly_solves)

