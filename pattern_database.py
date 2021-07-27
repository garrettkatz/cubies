import numpy as np

class PatternDatabase:
    def __init__(self, patterns, macros):
        self.patterns = patterns
        self.macros = macros
        self.match_index = 0
        self.matched = False
    def query(self, state):

        # unify state with patterns
        mask = np.ones(self.patterns.shape[0], dtype=bool)
        for k in range(len(state)):
            mask &= (state[k] == self.patterns[:,k]) | (0 == self.patterns[:,k]) # need to variable-bind
        
        self.match_index = mask.argmax() # what about multiple matches?
        self.matched = (mask[self.match_index] != False)

        return self.matched

    def result(self): # assumes match was not False
        return list(self.macros[self.match_index])

if __name__ == "__main__":

    from cube import CubeDomain
    domain = CubeDomain(3)
    descendent = domain.solved_state()

    pattern_database = PatternDatabase(descendent[np.newaxis,:].copy(), [((1,0,1),)])

    # Non-empty macro if state matches a database pattern
    matched = pattern_database.query(descendent)
    print(matched)
    print(pattern_database.result())


