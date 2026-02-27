"""AST-based safe arithmetic evaluator for derived variable expressions."""

from __future__ import annotations

import ast
import operator
from typing import Any

_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expr: str, variables: dict[str, float]) -> float:
    """Evaluate a simple arithmetic expression with named variables.

    Only supports ``+  -  *  /  //  %  **`` and the builtins
    ``round, abs, min, max``.  No attribute access, no imports.
    """
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise NameError(f"undefined variable: {node.id}")

    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported op: {ast.dump(node.op)}")
        return op(_eval_node(node.left, env), _eval_node(node.right, env))

    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {ast.dump(node.op)}")
        return op(_eval_node(node.operand, env))

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [_eval_node(a, env) for a in node.args]
            return float(_FUNCS[node.func.id](*args))
        raise ValueError(f"unsupported function: {ast.dump(node.func)}")

    raise ValueError(f"unsupported AST node: {ast.dump(node)}")
