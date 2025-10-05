import pytest
from typing import Set
from src.cli.filter_param import (
    FilterNode,
    TagNode,
    AndNode,
    OrNode,
    NotNode,
    FilterExpression,
)

@pytest.fixture
def tags_simple() -> Set[str]:
    """A standard set of tags for testing matches."""
    return {"vns", "tabu", "60s"}

# ====================================================================
# TEST 1: TagNode
# ====================================================================

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

# ====================================================================
# TEST 2: AndNode
# ====================================================================

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
    assert node.is_match({"any"}) # vacuously true

# ====================================================================
# TEST 3: OrNode
# ====================================================================

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

# ====================================================================
# TEST 4: Nested Logic (Combining Nodes)
# ====================================================================

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

# ====================================================================
# TEST 5: FilterExpression End-to-End
# ====================================================================

@pytest.mark.parametrize("config_name, root_node, expected", [
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
])
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


def test_andnode_logic(tags_simple):
    assert AndNode([TagNode("vns"), TagNode("60s")]).is_match(tags_simple)
    assert not AndNode([TagNode("vns"), TagNode("120s")]).is_match(tags_simple)

def test_ornode_logic(tags_simple):
    assert OrNode([TagNode("vns"), TagNode("120s")]).is_match(tags_simple)
    assert not OrNode([TagNode("nsga2"), TagNode("120s")]).is_match(tags_simple)


def test_filterexpression_with_not():
    """Test end-to-end filtering using a NOT node."""
    # Config Name: "vns 60s"
    # Filter Tree: NOT(tabu) AND 60s

    # Create the filter tree:
    not_tabu = NotNode(TagNode("tabu"))
    root_node = AndNode([not_tabu, TagNode("60s")])

    expression = FilterExpression(root_node)

    # Test 1: Config has 'tabu' and '60s' -> (NOT(True) AND True) -> False
    assert not expression.is_match("vns tabu 60s")

    # Test 2: Config has 'vns' and '60s' but NOT 'tabu' -> (NOT(False) AND True) -> True
    assert expression.is_match("vns 60s")
