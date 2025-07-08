"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the input unchanged.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate a number.
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Check if x is less than y.
    Returns 1.0 if x < y, 0.0 otherwise.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Check if x is equal to y.
    Returns 1.0 if x == y, 0.0 otherwise.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Return the maximum of x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Check if x and y are close in value.
    Returns 1.0 if |x - y| < 1e-2, 0.0 otherwise.
    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function.
    For x >= 0: f(x) = 1.0 / (1.0 + e^(-x))
    For x < 0: f(x) = e^x / (1.0 + e^x)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x: float) -> float:
    """
    Apply the ReLU activation function.
    Returns x if x > 0, 0.0 otherwise.
    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """
    Calculate the natural logarithm of x.
    """
    return math.log(x)


def exp(x: float) -> float:
    """
    Calculate the exponential function e^x.
    """
    return math.exp(x)


def inv(x: float) -> float:
    """
    Calculate the reciprocal of x (1/x).
    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """
    Compute the derivative of log(x) times d.
    The derivative of log(x) is 1/x, so this returns d/x.
    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """
    Compute the derivative of inv(x) times d.
    The derivative of 1/x is -1/x^2, so this returns -d/x^2.
    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    Compute the derivative of relu(x) times d.
    The derivative of relu(x) is 1 if x > 0, 0 otherwise.
    So this returns d if x > 0, 0 otherwise.
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> List[float]:
    """
    Higher-order map function.
    
    Args:
        fn: Function to apply to each element
        ls: Iterable of elements
        
    Returns:
        List with each element transformed by fn
    """
    return [fn(x) for x in ls]


def zipWith(fn: Callable[[float, float], float], 
           ls1: Iterable[float], 
           ls2: Iterable[float]) -> List[float]:
    """
    Higher-order zipWith function.
    
    Args:
        fn: Binary function to apply to pairs of elements
        ls1: First iterable
        ls2: Second iterable
        
    Returns:
        List with elements combined using fn
    """
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(fn: Callable[[float, float], float], 
          ls: Iterable[float], 
          start: float = 0.0) -> float:
    """
    Higher-order reduce function.
    
    Args:
        fn: Binary function to combine elements
        ls: Iterable of elements
        start: Starting value for the accumulation
        
    Returns:
        Single value after combining all elements
    """
    result = start
    for x in ls:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> List[float]:
    """
    Negate all elements in a list.
    
    Args:
        ls: Iterable of elements
        
    Returns:
        List with all elements negated
    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    """
    Add corresponding elements from two lists.
    
    Args:
        ls1: First iterable
        ls2: Second iterable
        
    Returns:
        List with elements added pairwise
    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """
    Sum all elements in a list.
    
    Args:
        ls: Iterable of elements
        
    Returns:
        Sum of all elements
    """
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """
    Calculate the product of all elements in a list.
    
    Args:
        ls: Iterable of elements
        
    Returns:
        Product of all elements
    """
    return reduce(mul, ls, 1.0)
