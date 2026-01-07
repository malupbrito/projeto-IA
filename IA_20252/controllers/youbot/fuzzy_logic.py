import numpy as np

# Funções de pertinência
def triangular(a, b, c):
    def func(x):
        if a == b and x == a:  # pico degenerado
            return 1.0
        if b == c and x == b:  # pico degenerado
            return 1.0

        if x <= a or x >= c:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if (b - a) != 0 else 0.0
        elif b <= x < c:
            return (c - x) / (c - b) if (c - b) != 0 else 0.0
        else:
            return 0.0
    return func


def trapezoidal(a, b, c, d):
    def func(x):
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if (b - a) != 0 else 0.0
        elif b <= x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c) if (d - c) != 0 else 0.0
        else:
            return 0.0
    return func


def defuzzify(universe, membership_values, method="centroid"):
    universe = np.array(universe)
    membership_values = np.array(membership_values)
    return np.sum(universe * membership_values) / (np.sum(membership_values) + 0.0000001)

# Classes
class FuzzySet:
    def __init__(self, name, func):
        self.name = name
        self.func = func
    def membership(self, x):
        return self.func(x)

class LinguisticVariable:
    def __init__(self, name, universe):
        self.name = name
        self.universe = universe
        self.sets = {}
    def add_set(self, fuzzy_set):
        self.sets[fuzzy_set.name] = fuzzy_set
    def fuzzify(self, x):
        return {name: f.membership(x) for name, f in self.sets.items()}

class Rule:
    def __init__(self, antecedent, consequent, operator="AND"):
        self.antecedent = antecedent
        self.consequent = consequent
        self.operator = operator
    def evaluate(self, fuzzified_inputs):
        values = []
        for var, set_name in self.antecedent.items():
            values.append(fuzzified_inputs[var][set_name])
        truth_value = min(values) if self.operator == "AND" else max(values)
        return {var: (set_name, truth_value) for var, set_name in self.consequent.items()}

class FuzzySystem:
    def __init__(self):
        self.variables = {}
        self.rules = []
    def add_variable(self, variable):
        self.variables[variable.name] = variable
    def add_rule(self, rule):
        self.rules.append(rule)
    def infer(self, inputs):
        fuzzified = {var: self.variables[var].fuzzify(val) for var, val in inputs.items()}
        results = []
        for rule in self.rules:
            results.append(rule.evaluate(fuzzified))
        return results
