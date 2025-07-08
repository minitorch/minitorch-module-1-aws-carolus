from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # equation: (f(..., x_i + h, ...) - f(..., x_i - h, ...)) / 2h
    x_i = vals[arg]
    h = epsilon
    x_i_plus_h = x_i + h
    x_i_minus_h = x_i - h
    f_plus_h = f(*vals[:arg], x_i_plus_h, *vals[arg + 1 :])
    f_minus_h = f(*vals[:arg], x_i_minus_h, *vals[arg + 1 :])
    return (f_plus_h - f_minus_h) / (2 * h)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result = []
    
    def visit(var: Variable) -> None:
        # Skip if already visited or if it's a constant
        if var.unique_id in visited or var.is_constant():
            return
        
        # Append this before its parents since we want to return list from the right
        visited.add(var.unique_id)
        
        # Visit all parents (inputs)
        for parent in var.parents:
            visit(parent)

        result.append(var)
    
    # Start DFS from the rightmost variable
    visit(variable)
    
    # Return in reverse order (from right to left)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Get variables in topological order
    sorted_variables = topological_sort(variable)
    
    # Dictionary to store derivatives for each variable
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    
    # Process each variable in reverse topological order
    for var in sorted_variables:
        # Skip if this variable doesn't have a derivative yet
        if var.unique_id not in derivatives:
            continue
            
        current_deriv = derivatives[var.unique_id]
        
        # If this is a leaf node, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(current_deriv)
        
        # Propagate the derivative to parents using chain rule
        if not var.is_leaf():
            for parent_var, parent_deriv in var.chain_rule(current_deriv):
                parent_id = parent_var.unique_id
                if parent_id not in derivatives:
                    derivatives[parent_id] = 0.0
                derivatives[parent_id] += parent_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
