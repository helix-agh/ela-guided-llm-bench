import numpy as np
from deap import gp


def create_pset():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addEphemeralConstant("real_num", lambda: np.random.random() * 9 + 1)
    pset.addPrimitive(first_dv, 1)
    pset.addPrimitive(trans_dv, 1)
    pset.addEphemeralConstant("rand_num", lambda: 1 + np.random.random() / 10)
    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(neg, 1)
    pset.addPrimitive(reciprocal, 1)
    pset.addPrimitive(mul10, 1)
    pset.addPrimitive(square, 1)
    pset.addPrimitive(sqrt, 1)
    pset.addPrimitive(abs_, 1)
    pset.addPrimitive(roundoff, 1)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(ln, 1)
    pset.addPrimitive(exp, 1)
    pset.addPrimitive(sum_vec, 1)
    pset.addPrimitive(mean_vec, 1)
    pset.addPrimitive(cumsum_vec, 1)
    pset.addPrimitive(prod_vec, 1)
    pset.addPrimitive(amax_vec, 1)
    pset.renameArguments(ARG0="x")
    return pset


def first_dv(x):
    if np.isscalar(x):
        return x
    return x[0]


def trans_dv(x):
    if np.isscalar(x):
        return x
    return np.hstack((x[1:].ravel(), np.zeros((1, 1)).ravel()))


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return np.where(np.abs(y) > 1e-20, np.divide(x, y), 1.0)


def neg(x):
    return -1 * x


def reciprocal(x):
    return np.where(np.abs(x) > 1e-20, np.divide(1, x), 1.0)


def mul10(x):
    return 10 * x


def square(x):
    return np.square(x)


def sqrt(x):
    return np.sqrt(abs(x))


def abs_(x):
    return abs(x)


def roundoff(x):
    return np.round(x)


def sin(x):
    return np.sin(2 * np.pi * x)


def cos(x):
    return np.cos(2 * np.pi * x)


def ln(x):
    return np.where(np.abs(x) > 1e-20, np.log(abs(x)), 1.0)


def exp(x):
    return np.exp(x)


def sum_vec(x):
    return np.sum(x)


def mean_vec(x):
    return np.mean(x)


def cumsum_vec(x):
    return np.cumsum(x)


def prod_vec(x):
    return np.prod(x)


def amax_vec(x):
    return np.amax(x)
