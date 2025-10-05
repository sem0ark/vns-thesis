import re
from typing import Any, Optional

import click


class FilterNode:
    """Abstract base class for a node in the filter expression tree."""

    def is_match(self, config_tags: set[str]) -> bool:
        """
        Evaluates the node against a set of configuration tags.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class TagNode(FilterNode):
    """Represents a single tag (leaf node) in the expression."""

    def __init__(self, tag: str):
        self.tag = tag.lower()

    def is_match(self, config_tags: set[str]) -> bool:
        """A tag matches if it is present in the set of config tags."""
        return self.tag in config_tags


class AndNode(FilterNode):
    """Represents an 'AND' operation (internal node)."""

    def __init__(self, children: list[FilterNode]):
        self.children = children

    def is_match(self, config_tags: set[str]) -> bool:
        """AND matches if all child nodes match."""
        return all(child.is_match(config_tags) for child in self.children)


class OrNode(FilterNode):
    """Represents an 'OR' operation (internal node)."""

    def __init__(self, children: list[FilterNode]):
        self.children = children

    def is_match(self, config_tags: set[str]) -> bool:
        """OR matches if at least one child node matches."""
        return any(child.is_match(config_tags) for child in self.children)


class NotNode(FilterNode):
    """Represents an 'NOT' operation (internal node)."""

    def __init__(self, child: FilterNode):
        self.child = child

    def is_match(self, config_tags: set[str]) -> bool:
        """NOT matches if at child node does not match."""
        return not self.child.is_match(config_tags)


class FilterExpression:
    """The result object returned by the Click custom type."""

    def __init__(self, root_node: FilterNode | None, initial_string: str = ""):
        self.root = root_node
        self.initial_string = initial_string

    def is_match(self, config_name: str) -> bool:
        """
        Prepares the config name and evaluates the parsed expression tree.
        Receives the raw config_name and prepares it (lowercasing, splitting).
        """
        if self.root is None:
            return True  # Empty filter matches everything

        # Prepare the tags once
        config_tags = set(config_name.strip().lower().split())

        return self.root.is_match(config_tags)

    def __repr__(self) -> str:
        return f"Filter({self.initial_string.lower()})"


def _parse_expression(tokens: list[FilterNode | str]) -> FilterNode:
    """
    Parses a token list into a FilterNode tree respecting AND/OR precedence.
    This uses a simplified operator-precedence parsing approach.
    """
    if not tokens:
        raise ValueError("Malformed filter expression: Empty expression.")

    # Resolve 'NOT' operations
    i = 0
    while i < len(tokens):
        if tokens[i] == "not":
            if i == len(tokens) - 1:
                raise ValueError(
                    "Malformed filter expression: 'not' requires an operand."
                )

            op1 = tokens[i + 1]
            if not isinstance(op1, FilterNode):
                # Ensure 'not' isn't followed by another operator (e.g., 'not and')
                raise ValueError(
                    f"Malformed filter expression: Expected tag/group, found operator '{op1}' after 'not'."
                )

            new_node = NotNode(op1)
            tokens = tokens[:i] + [new_node] + tokens[i + 2 :]
            i = 0
        else:
            i += 1

    # Resolve 'AND' operations
    i = 0
    while i < len(tokens):
        if tokens[i] == "and":
            if i == 0 or i == len(tokens) - 1:
                raise ValueError(
                    "Malformed filter expression: 'and' requires operands."
                )

            op1 = tokens[i - 1]
            if not isinstance(op1, FilterNode):
                raise ValueError(
                    f"Malformed filter expression: Expected tag/group, found operator '{op1}'."
                )

            op2 = tokens[i + 1]
            if not isinstance(op2, FilterNode):
                raise ValueError(
                    f"Malformed filter expression: Expected tag/group, found operator '{op2}'."
                )

            new_node = AndNode([op1, op2])
            tokens = tokens[: i - 1] + [new_node] + tokens[i + 2 :]
            i = 0  # Restart scanning for 'and'
        else:
            i += 1

    # Resolve 'OR' operations (lower precedence)
    # The remaining structure should be Node - 'or' - Node - 'or' - Node ...
    if len(tokens) == 1:
        if not isinstance(tokens[0], FilterNode):
            raise ValueError(
                f"Malformed filter expression: Unhandled token '{tokens[0]}'."
            )
        return tokens[0]

    if len(tokens) % 2 != 1:
        raise ValueError(
            "Malformed filter expression: Operators missing between terms."
        )

    # Group all remaining nodes under a single OrNode
    or_children = []
    i = 0
    while i < len(tokens):
        node = tokens[i]
        if not isinstance(node, FilterNode):
            raise ValueError(
                f"Malformed filter expression: Expected tag/group, found operator '{node}'."
            )
        or_children.append(node)
        i += 2

    return OrNode(or_children)


def _tokenize_and_parse(expression: str) -> FilterNode:
    """
    Splits the expression into a list of tokens, resolving parentheses recursively.
    Returns a flattened list of tags, AND/OR keywords, and the root node of any sub-expression.
    """
    tokens = re.findall(r"\(|\)|\band\b|\bor\b|[^()\s]+", expression.lower())
    tokens_processed: list[FilterNode | str] = []

    for token in tokens:
        if isinstance(token, str):
            token_lower = token.strip().lower()

            if token_lower in ("(", ")", "and", "or", "not"):
                tokens_processed.append(token_lower)
            elif token_lower:
                tokens_processed.append(TagNode(token_lower))
        else:
            raise ValueError(f"Unreachable: {token}")

    while "(" in tokens_processed:
        try:
            open_idx = [i for i, t in enumerate(tokens_processed) if t == "("][-1]
            close_idx = tokens_processed[open_idx:].index(")") + open_idx
        except (IndexError, ValueError):
            raise ValueError("Malformed filter expression: Unmatched parenthesis.")

        sub_expression_tokens = tokens_processed[open_idx + 1 : close_idx]
        sub_node = _parse_expression(sub_expression_tokens)
        tokens_processed = (
            tokens_processed[:open_idx] + [sub_node] + tokens_processed[close_idx + 1 :]
        )

    return _parse_expression(tokens_processed)


class ClickFilterExpression(click.ParamType):
    """
    A custom Click parameter type that parses a Boolean filter expression
    and returns a FilterExpression object with an is_match method.
    """

    name = "filter_expression"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> FilterExpression:
        if not isinstance(value, str) or not value.strip():
            # Returns an expression that matches everything if filter is empty
            return FilterExpression(None, "")

        expression = value.strip()

        try:
            root_node = _tokenize_and_parse(expression)
            return FilterExpression(root_node, expression)
        except ValueError as e:
            raise click.BadParameter(f"Invalid filter expression syntax: {e}")


# def common_options(f):
#     """Apply common CLI options for both run and show actions."""

#     f = click.option(
#         "-i",
#         "--instance",
#         required=True,
#         type=str,
#         help="Path pattern (with wildcards) to instance files (for run) or single path (for show).",
#     )(f)

#     f = click.option(
#         "-f",
#         "--filter-string",
#         default="",
#         type=ClickFilterExpression(),
#         help="Boolean filter expression for config names (e.g., '(vns or nsga2) and 120s').",
#     )(f)

#     return f


# def _execute_run_logic(self, instance: str, max_time: str, filter_expression: FilterExpression):
#     """Contains the logic for running optimizations and saving results."""
#     # ... (rest of the logic remains the same) ...

#     for instance_path_str in instance_paths:
#         configuration = RunConfig(run_time_seconds, Path(instance_path_str))

#         problem_configs = {
#             config_name: func
#             for runner in self.runners
#             for config_name, func in runner(configuration).get_variants()
#             # *** Use the is_match method here ***
#             if filter_expression.is_match(config_name)
#         }
#         # ... (rest of the logic) ...

# # The other execution methods (_execute_plot_logic, _execute_show_logic) must be updated similarly
# # to receive `filter_expression: FilterExpression` instead of `filter_string: str` and use `if filter_expression.is_match(config_name):`
