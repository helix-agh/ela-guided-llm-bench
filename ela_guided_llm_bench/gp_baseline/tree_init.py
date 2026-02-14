import random
import sys
from inspect import isclass

import numpy as np
from deap import gp


def genFull_(points, pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height

    return generate_(points, pset, min_, max_, condition, type_)


def genGrow_(points, pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)

    return generate_(points, pset, min_, max_, condition, type_)


def genHalfAndHalf_(points, pset, min_, max_, type_=None):
    method = random.choice((genGrow_, genFull_))
    return method(points, pset, min_, max_, type_)


def generate_(points, pset, min_, max_, condition, type_=None):
    while True:
        expr = generate_base(pset, min_, max_, condition, type_=type_)
        f_ = gp.compile(gp.PrimitiveTree(expr), pset)
        try:
            list_y = []
            for i in range(len(points)):
                y = f_(points[i])
                list_y.append(np.mean(y))
            y = np.array(list_y)
            y[abs(y) < 1e-20] = 0.0
        except Exception:
            continue
        if np.isnan(y).any() or np.isinf(y).any() or np.var(y) < 1e-20:
            continue
        break
    return expr


def generate_base(pset, min_, max_, condition, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]

    pTerminal = np.array([10, 5, 1])
    pTerminal = np.cumsum(pTerminal)
    pTerminal = pTerminal / np.max(pTerminal)
    assert len(pset.terminals[type_]) == len(pTerminal)

    pPrimitive = [1, 1, 15, 15, 10, 10, 2, 2, 2, 5, 5, 2, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1]
    pPrimitive = np.cumsum(pPrimitive)
    pPrimitive = pPrimitive / np.max(pPrimitive)
    assert len(pset.primitives[type_]) == len(pPrimitive)

    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = pset.terminals[type_][np.argwhere(random.random() <= pTerminal)[0][0]]
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = pset.primitives[type_][np.argwhere(random.random() <= pPrimitive)[0][0]]
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr
