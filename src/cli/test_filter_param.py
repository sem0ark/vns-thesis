from typing import Set

import pytest

from src.cli.filter_param import (
    AndNode,
    FilterExpression,
    FilterNode,
    NotNode,
    OrNode,
    TagNode,
    _parse_expression,
    _tokenize_and_parse,
)


@pytest.fixture
def tags_simple() -> Set[str]:
    """A standard set of tags for testing matches."""
    return {"vns", "tabu", "60s"}


def test_tagnode_match_found(tags_simple):
    """Test TagNode when the tag is present."""
    node = TagNode("vns")
    assert node.is_match(tags_simple)


def test_tagnode_match_not_found(tags_simple):
    """Test TagNode when the tag is absent."""
    node = TagNode("nsga2")
    assert not node.is_match(tags_simple)


def test_tagnode_case_insensitivity():
    """Test TagNode ensures case-insensitivity during comparison."""
    tags = {"vns", "tabu"}
    node_lower = TagNode("vns")
    node_upper = TagNode("TABU")
    assert node_lower.is_match(tags)
    assert node_upper.is_match(tags)


def test_andnode_all_match(tags_simple):
    """Test AndNode where all children match (True AND True)."""
    node = AndNode([TagNode("vns"), TagNode("60s")])
    assert node.is_match(tags_simple)


def test_andnode_one_fails(tags_simple):
    """Test AndNode where one child fails (True AND False)."""
    node = AndNode([TagNode("vns"), TagNode("120s")])
    assert not node.is_match(tags_simple)


def test_andnode_empty_children():
    """Test AndNode with no children (should evaluate to True)."""
    node = AndNode([])
    assert node.is_match({"any"})  # vacuously true


def test_ornode_both_match(tags_simple):
    """Test OrNode where both children match (True OR True)."""
    node = OrNode([TagNode("vns"), TagNode("60s")])
    assert node.is_match(tags_simple)


def test_ornode_one_matches(tags_simple):
    """Test OrNode where one child matches (True OR False)."""
    node = OrNode([TagNode("vns"), TagNode("120s")])
    assert node.is_match(tags_simple)


def test_ornode_none_match(tags_simple):
    """Test OrNode where no children match (False OR False)."""
    node = OrNode([TagNode("nsga2"), TagNode("120s")])
    assert not node.is_match(tags_simple)


def test_ornode_empty_children():
    """Test OrNode with no children (should evaluate to False)."""
    node = OrNode([])
    assert not node.is_match({"any"})


def test_nested_complex_match(tags_simple):
    """Test a complex nested structure that should match."""
    # Expression: (vns OR 120s) AND tabu
    or_clause = OrNode([TagNode("vns"), TagNode("120s")])
    root = AndNode([or_clause, TagNode("tabu")])

    # (True OR False) AND True -> True
    assert root.is_match(tags_simple)


def test_nested_complex_fail(tags_simple):
    """Test a complex nested structure that should fail."""
    # Expression: (vns AND 120s) OR tabu -> (False) OR True -> True (Oops, need one that fails)
    # Expression: (vns AND 120s) OR nsga2
    and_clause = AndNode([TagNode("vns"), TagNode("120s")])
    root = OrNode([and_clause, TagNode("nsga2")])

    # (True AND False) OR False -> False
    assert not root.is_match(tags_simple)


@pytest.mark.parametrize(
    "config_name, root_node, expected",
    [
        # Simple match
        ("vns 60s tabu", TagNode("vns"), True),
        # Simple fail
        ("vns 60s tabu", TagNode("nsga2"), False),
        # Match with different casing/spacing
        (" VNS  60s Tabu ", TagNode("tabu"), True),
        # Nested match
        ("vns 120s", AndNode([TagNode("vns"), TagNode("120s")]), True),
        # Nested fail
        ("vns 120s", AndNode([TagNode("vns"), TagNode("60s")]), False),
    ],
)
def test_filterexpression_is_match(config_name, root_node, expected):
    """Tests the public is_match interface, including name preparation."""
    expression = FilterExpression(root_node)
    assert expression.is_match(config_name) == expected


def test_filterexpression_empty_root():
    """Test FilterExpression returns True when the filter is empty (root is None)."""
    expression = FilterExpression(None)
    assert expression.is_match("any config name")


def test_filterexpression_abstract_error():
    """Test that FilterNode raises an error if used directly."""
    with pytest.raises(NotImplementedError):
        FilterNode().is_match({"a"})


def test_notnode_negates_match(tags_simple):
    """Test NOT(vns) should be False if 'vns' is present."""
    node = NotNode(TagNode("vns"))
    assert not node.is_match(tags_simple)


def test_notnode_negates_non_match(tags_simple):
    """Test NOT(nsga2) should be True if 'nsga2' is absent."""
    node = NotNode(TagNode("nsga2"))
    assert node.is_match(tags_simple)


def test_notnode_nested_negation(tags_simple):
    """Test NOT(NOT(vns)) is equivalent to 'vns'."""
    inner_not = NotNode(TagNode("vns"))
    outer_not = NotNode(inner_not)
    assert outer_not.is_match(tags_simple)


def test_notnode_negating_compound(tags_simple):
    """Test NOT(vns AND 120s) where vns is present, 120s is absent."""
    # (vns AND 120s) -> (True AND False) -> False
    # NOT(False) -> True
    and_node = AndNode([TagNode("vns"), TagNode("120s")])
    node = NotNode(and_node)
    assert node.is_match(tags_simple)


@pytest.fixture
def nodes() -> list[TagNode]:
    """Fixture returning simple TagNode objects."""
    return [TagNode(f"tag{i}") for i in range(5)]


def test_parse_simple_tag(nodes):
    """Test parsing a single node."""
    result = _parse_expression([nodes[0]])
    assert result == nodes[0]
    assert isinstance(result, TagNode)


def test_parse_and_precedence(nodes):
    """Test AND (higher precedence) is grouped correctly."""
    # Input: [tag0, 'or', tag1, 'and', tag2]
    tokens = [nodes[0], "or", nodes[1], "and", nodes[2]]
    result = _parse_expression(tokens)

    # Expected: OrNode([tag0, AndNode([tag1, tag2])])
    assert isinstance(result, OrNode)
    assert len(result.children) == 2
    assert result.children[0] == nodes[0]  # tag0 is the first OR child

    and_node = result.children[1]
    assert isinstance(and_node, AndNode)
    assert and_node.children[0] == nodes[1]
    assert and_node.children[1] == nodes[2]


def test_parse_or_grouping(nodes):
    """Test multiple OR operators group correctly under one OrNode."""
    # Input: [tag0, 'or', tag1, 'or', tag2]
    tokens = [nodes[0], "or", nodes[1], "or", nodes[2]]
    result = _parse_expression(tokens)

    # Expected: OrNode([tag0, tag1, tag2])
    assert isinstance(result, OrNode)
    assert len(result.children) == 3
    assert all(isinstance(c, TagNode) for c in result.children)


def test_parse_complex_precedence(nodes):
    """Test a mix of AND and OR."""
    # Input: [tag0, 'and', tag1, 'or', tag2, 'and', tag3]
    tokens = [nodes[0], "and", nodes[1], "or", nodes[2], "and", nodes[3]]
    result = _parse_expression(tokens)

    # Expected: OrNode([AndNode([tag0, tag1]), AndNode([tag2, tag3])])
    assert isinstance(result, OrNode)
    assert len(result.children) == 2
    assert isinstance(result.children[0], AndNode)
    assert isinstance(result.children[1], AndNode)


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["and", TagNode("a")],
        [TagNode("a"), "or"],
        [TagNode("a"), "or", "and", TagNode("b")],
        [TagNode("a"), TagNode("b")],
    ],
)
def test_parse_expression_malformed_errors(tokens):
    """Test various malformed token lists raise appropriate errors."""
    with pytest.raises(ValueError):
        _parse_expression(tokens)


def test_tokenize_and_parse_simple_and():
    """Test full parsing for a simple expression without parentheses."""
    # Expression: "vns and 60s"
    result = _tokenize_and_parse("vns and 60s")

    # Expected: AndNode([TagNode('vns'), TagNode('60s')])
    assert isinstance(result, AndNode)
    assert result.children[0].tag == "vns"
    assert result.children[1].tag == "60s"


def test_tokenize_and_parse_parentheses_override(nodes):
    """Test parentheses override precedence."""
    # Expression: (tag0 or tag1) and tag2
    expr = "(tag0 or tag1) and tag2"
    result = _tokenize_and_parse(expr)

    # Expected: AndNode([OrNode([tag0, tag1]), tag2])
    assert isinstance(result, AndNode)
    or_node = result.children[0]
    assert isinstance(or_node, OrNode)
    assert or_node.children[0].tag == "tag0"
    assert result.children[1].tag == "tag2"


def test_tokenize_and_parse_nested_parentheses():
    """Test deeply nested parentheses."""
    # Expression: (tag0 and (tag1 or tag2))
    expr = "(tag0 and (tag1 or tag2))"
    result = _tokenize_and_parse(expr)

    # Expected: AndNode([tag0, OrNode([tag1, tag2])])
    assert isinstance(result, AndNode)
    assert result.children[0].tag == "tag0"
    assert isinstance(result.children[1], OrNode)
    assert result.children[1].children[0].tag == "tag1"


def test_tokenize_and_parse_case_and_whitespace():
    """Test case-insensitivity and whitespace handling."""
    # Expression: " VNS OR (  nsga2 AnD 120s ) "
    expr = " VNS OR (  nsga2 AnD 120s ) "
    result = _tokenize_and_parse(expr)

    # Expected: OrNode([TagNode('vns'), AndNode([TagNode('nsga2'), TagNode('120s')])])
    assert isinstance(result, OrNode)
    assert result.children[0].tag == "vns"
    and_node = result.children[1]
    assert and_node.children[0].tag == "nsga2"
    assert and_node.children[1].tag == "120s"


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("a and (b or c"),  # Unmatched parenthesis (unclosed)
        pytest.param("(a and b) or c)"),  # Unmatched parenthesis (unopened)
        pytest.param("and a or b"),
        pytest.param("a b"),
    ],
)
def test_tokenize_and_parse_malformed_errors(expression):
    """Test parsing failures due to parentheses or final structure."""
    with pytest.raises(ValueError):
        _tokenize_and_parse(expression)


def test_tokenize_and_parse_simple_not():
    """Test full parsing for a simple NOT expression."""
    # Expression: "not vns"
    result = _tokenize_and_parse("not vns")

    # Expected: NotNode(TagNode('vns'))
    assert isinstance(result, NotNode)
    assert isinstance(result.child, TagNode)
    assert result.child.tag == "vns"


def test_tokenize_and_parse_not_and():
    """Test NOT applied to an AND expression (highest precedence)."""
    # Expression: "not vns and 60s" -> (NOT vns) AND 60s
    # NOT has higher precedence than AND, so it should apply only to the immediate tag.
    result = _tokenize_and_parse("not vns and 60s")

    # Expected: AndNode([NotNode(TagNode('vns')), TagNode('60s')])
    assert isinstance(result, AndNode)
    assert isinstance(result.children[0], NotNode)
    assert result.children[0].child.tag == "vns"
    assert result.children[1].tag == "60s"


def test_tokenize_and_parse_not_in_parentheses():
    """Test NOT applied within parentheses to change precedence."""
    # Expression: "not (vns and 60s)"
    result = _tokenize_and_parse("not (vns and 60s)")

    # Expected: NotNode(AndNode([TagNode('vns'), TagNode('60s')]))
    assert isinstance(result, NotNode)
    assert isinstance(result.child, AndNode)
    assert result.child.children[0].tag == "vns"


def test_tokenize_and_parse_not_case_insensitivity():
    """Test NOT operator is case-insensitive."""
    # Expression: "NOT tabu OR nOt 120s"
    result = _tokenize_and_parse("NOT tabu OR nOt 120s")

    # Expected: OrNode([NotNode(TagNode('tabu')), NotNode(TagNode('120s'))])
    assert isinstance(result, OrNode)
    assert isinstance(result.children[0], NotNode)
    assert isinstance(result.children[1], NotNode)
    assert result.children[0].child.tag == "tabu"
    assert result.children[1].child.tag == "120s"


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("not"),  # Missing operand
        pytest.param("and not vns"),  # Operator sequence (and followed by not)
        pytest.param("not and vns"),  # Operator sequence (not followed by and)
    ],
)
def test_tokenize_and_parse_not_malformed_errors(expression):
    """Test parsing failures when NOT is misused."""
    # Note: These tests depend on the ValueError checks inside _parse_expression.
    with pytest.raises(ValueError):
        _tokenize_and_parse(expression)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("not not vns"),  # TODO: allow not not parsing logic
    ],
)
def test_tokenize_and_parse_expressions(expression):
    """Test parsing failures when NOT is misused."""
    # Note: These tests depend on the ValueError checks inside _parse_expression.
    with pytest.raises(ValueError):
        _tokenize_and_parse(expression)
