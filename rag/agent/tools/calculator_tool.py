from __future__ import annotations

import ast
import operator
from typing import Any, Callable


Number = int | float


# 允许使用的二元运算符。
_BINARY_OPERATORS: dict[
    type[ast.operator],
    Callable[[Number, Number], Number],
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


# 允许使用的一元运算符。
_UNARY_OPERATORS: dict[
    type[ast.unaryop],
    Callable[[Number], Number],
] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 10
MAX_ABS_RESULT = 1_000_000_000_000_000


class SafeCalculator(ast.NodeVisitor):
    """
    Evaluate a limited arithmetic expression safely.

    Supported:
    +, -, *, /, //, %, **, parentheses

    Not supported:
    variables, function calls, imports, attribute access, lists, dictionaries
    """

    def visit_Expression(self, node: ast.Expression) -> Number:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Number:
        value = node.value

        # bool is a subclass of int in Python, so block it explicitly.
        if isinstance(value, bool):
            raise ValueError("Boolean values are not supported.")

        if not isinstance(value, (int, float)):
            raise ValueError("Only integer and floating-point numbers are allowed.")

        return value

    def visit_BinOp(self, node: ast.BinOp) -> Number:
        operator_type = type(node.op)

        if operator_type not in _BINARY_OPERATORS:
            raise ValueError(
                f"Unsupported operator: {operator_type.__name__}"
            )

        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(
                f"Exponent must be between {-MAX_EXPONENT} "
                f"and {MAX_EXPONENT}."
            )

        operation = _BINARY_OPERATORS[operator_type]
        result = operation(left, right)

        return self._validate_result(result)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Number:
        operator_type = type(node.op)

        if operator_type not in _UNARY_OPERATORS:
            raise ValueError(
                f"Unsupported unary operator: {operator_type.__name__}"
            )

        operand = self.visit(node.operand)
        operation = _UNARY_OPERATORS[operator_type]
        result = operation(operand)

        return self._validate_result(result)

    def generic_visit(self, node: ast.AST) -> Number:
        """
        Reject every AST node that was not explicitly allowed above.
        """

        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}"
        )

    @staticmethod
    def _validate_result(result: Any) -> Number:
        if isinstance(result, bool):
            raise ValueError("Boolean results are not supported.")

        if not isinstance(result, (int, float)):
            raise ValueError("The calculation did not produce a real number.")

        if abs(result) > MAX_ABS_RESULT:
            raise ValueError("The result is too large.")

        return result


def calculate_expression(expression: str) -> Number:
    """
    Parse and evaluate a safe arithmetic expression.
    """

    cleaned_expression = expression.strip()

    if not cleaned_expression:
        raise ValueError("The expression cannot be empty.")

    if len(cleaned_expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("The expression is too long.")

    try:
        syntax_tree = ast.parse(
            cleaned_expression,
            mode="eval",
        )
    except SyntaxError as exc:
        raise ValueError("The expression has invalid syntax.") from exc

    calculator = SafeCalculator()

    try:
        return calculator.visit(syntax_tree)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero is not allowed.") from exc


def format_number(value: Number) -> str:
    """
    Format 2.0 as 2 while keeping normal decimal values such as 2.5.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def run_calculator_tool(expression: str) -> dict[str, Any]:
    """
    Public tool interface used by the Agent Controller.
    """

    try:
        result = calculate_expression(expression)

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "answer": f"{expression} = {format_number(result)}",
            "error": None,
        }

    except ValueError as exc:
        return {
            "success": False,
            "expression": expression,
            "result": None,
            "answer": None,
            "error": str(exc),
        }