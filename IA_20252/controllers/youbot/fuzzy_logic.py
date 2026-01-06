def triangularFunction(x_top, x_left, x_right):
    def membership(x):
        return max(0.0, min((x - x_left) / (x_top - x_left), (x_right - x) / (x_right - x_top)))
    return membership

def trapezoidalFunction(x_top_a, x_top_b, x_left, x_right):
    def membership(x):
        return max(0.0, min((x - x_left) / (x_top_a - x_left), (x_right - x) / (x_right - x_top_b)))
    return membership

def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
         return 0.0
    elif b <= x <= c:
            return 1.0
    elif a < x < b:
         return (x - a) / (b - a)
    else:  
            return (d - x) / (d - c)


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)


def AND(a, b):
    return min(a, b)


   
def OR(a, b):
    return max(a, b)
    


def defuzzify(fuzzy_out, values):
    num = sum(fuzzy_out[k] * values[k] for k in fuzzy_out)
    den = sum(fuzzy_out[k] for k in fuzzy_out)
    return num / den if den > 0 else 0.0