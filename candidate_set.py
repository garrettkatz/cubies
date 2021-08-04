import numpy as np
from pattern_database import PatternDatabase
from algorithm import run

class Candidate:
    def __init__(self, patterns, macros):
        self.patterns = patterns
        self.macros = macros
        self.good_match_counts = None
        self.fail_match_counts = None
        self.match_counts = None

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
        candidate.good_match_counts = np.zeros(len(candidate.patterns), dtype=int)
        candidate.fail_match_counts = np.zeros(len(candidate.patterns), dtype=int)
        candidate.match_counts = np.zeros(len(candidate.patterns), dtype=int)

        ### Run candidate on problem instances
        pattern_database = PatternDatabase(candidate.patterns, candidate.macros, self.domain)
        result = {}
        scramble_length = {}
        plan_length = {}

        for i in range(self.num_instances):

            # Run algorithm on instance
            scramble_length[i] = self.rng.integers(1, self.rollout_length, endpoint=True)
            state = self.domain.random_state(scramble_length[i], self.rng)
            result[i] = run(state, self.domain, self.bfs_tree, pattern_database, self.max_depth, self.max_macros)
            success = (result[i] != False)

            # Record plan length
            if success:
                plan_length[i] = 0
                for (actions, sym, macro) in result[i]: plan_length[i] += len(actions) + len(macro)

            # Update traces
            if success:
                candidate.good_match_counts += pattern_database.match_counts
            else:
                candidate.fail_match_counts += pattern_database.match_counts
            pattern_database.reset()
    
        # Overall match counts
        candidate.match_counts = candidate.good_match_counts + candidate.fail_match_counts

        ### Evaluate objective functions
        # solve in <= 20 steps
        # solve in <= rollout_length steps
        # smaller number of patterns
        # smaller macro lengths
        # less complex patterns (like colors grouped together, more wildcards)?
        
        pattern_size = -len(candidate.patterns)
        macro_size = -sum(map(len, candidate.macros))
        godly_solves = sum(
            int((result[i] != False) and (plan_length[i] <= min(self.domain.god_number(), scramble_length[i])))
            for i in range(self.num_instances))
    
        objectives = (pattern_size, macro_size, godly_solves)

        return candidate, objectives

    def mutate_unmatched(self, candidate):
        patterns = list(candidate.patterns)
        macros = list(candidate.macros)
        match_counts = candidate.match_counts
        for p in np.flatnonzero(match_counts == match_counts.min()):
            patterns[p], macros[p] = self.sample_macro()
        return Candidate(patterns, macros)
    
if __name__ == "__main__":
    
    cube_size = 3
    max_scrambles = 5
    num_instances = 128
    tree_depth = 3
    max_depth = 1
    max_macros = 2
    num_patterns = 32
    min_macro_size = 1
    max_macro_size = 4
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

    candidate, objectives = candidate_set.evaluate(candidate_set.mutate_unmatched(candidate))
    pattern_size, macro_size, godly_solves = objectives

    print()
    print("num_patterns = %d" % num_patterns)
    print("macro_size = %d" % macro_size)
    print("godly_solves = %d" % godly_solves)

