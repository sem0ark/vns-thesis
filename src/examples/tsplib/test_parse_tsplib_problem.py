import math

from parse_tsplib_problem import TSPLibParser


def test_u1817_parsing():
    """
    Tests parsing of a typical TSP problem with NODE_COORD_SECTION.
    """
    file_content = """
NAME : u1817
COMMENT : Drilling problem (Reinelt)
TYPE : TSP
DIMENSION : 1817
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 6.51190e+02 2.24439e+03
2 6.76600e+02 2.16820e+03
3 7.01990e+02 2.16820e+03
4 7.27400e+02 2.16820e+03
5 7.52790e+02 2.16820e+03
6 7.27400e+02 2.14279e+03
EOF
    """
    parser = TSPLibParser()
    parsed_data = parser.parse_string(file_content)

    # Assert specification data
    assert parsed_data["specification"]["NAME"] == "u1817"
    assert parsed_data["specification"]["COMMENT"] == "Drilling problem (Reinelt)"
    assert parsed_data["specification"]["TYPE"] == "TSP"
    assert parsed_data["specification"]["DIMENSION"] == 1817
    assert parsed_data["specification"]["EDGE_WEIGHT_TYPE"] == "EUC_2D"

    # Assert node coordinates data
    expected_coords = {
        1: (651.190, 2244.390),
        2: (676.600, 2168.200),
        3: (701.990, 2168.200),
        4: (727.400, 2168.200),
        5: (752.790, 2168.200),
        6: (727.400, 2142.790),
    }
    assert "NODE_COORDS" in parsed_data["data"]
    # Check for approximate equality for floats due to potential precision differences
    for node_id, coords in expected_coords.items():
        assert node_id in parsed_data["data"]["NODE_COORDS"]
        assert math.isclose(
            parsed_data["data"]["NODE_COORDS"][node_id][0], coords[0], rel_tol=1e-9
        )
        assert math.isclose(
            parsed_data["data"]["NODE_COORDS"][node_id][1], coords[1], rel_tol=1e-9
        )
    assert len(parsed_data["data"]["NODE_COORDS"]) == len(expected_coords)


def test_full_matrix_edge_weights_parsing():
    """
    Tests parsing of a problem with an explicit FULL_MATRIX EDGE_WEIGHT_SECTION.
    """
    file_content_matrix = """
NAME : example_matrix
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1 2
1 0 3
2 3 0
EOF
    """
    parser = TSPLibParser()
    parsed_data = parser.parse_string(file_content_matrix)

    # Assert specification data
    assert parsed_data["specification"]["NAME"] == "example_matrix"
    assert parsed_data["specification"]["TYPE"] == "TSP"
    assert parsed_data["specification"]["DIMENSION"] == 3
    assert parsed_data["specification"]["EDGE_WEIGHT_TYPE"] == "EXPLICIT"
    assert parsed_data["specification"]["EDGE_WEIGHT_FORMAT"] == "FULL_MATRIX"

    # Assert edge weights data
    expected_matrix = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    assert "EDGE_WEIGHTS" in parsed_data["data"]
    actual_matrix = parsed_data["data"]["EDGE_WEIGHTS"]

    assert len(actual_matrix) == len(expected_matrix)
    for i in range(len(expected_matrix)):
        assert len(actual_matrix[i]) == len(expected_matrix[i])
        for j in range(len(expected_matrix[i])):
            assert math.isclose(
                actual_matrix[i][j], expected_matrix[i][j], rel_tol=1e-9
            )


def test_cvrp_parsing_with_demands_and_depots():
    """
    Tests parsing of a CVRP problem including NODE_COORD_SECTION, DEMAND_SECTION, and DEPOT_SECTION.
    """
    file_content_cvrp = """
NAME : P-n16-k8
TYPE : CVRP
DIMENSION : 16
CAPACITY : 100
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 40 50
2 20 60
3 70 80
4 10 20
5 30 10
6 80 40
7 60 20
8 90 70
9 50 30
10 20 40
11 80 10
12 50 60
13 70 30
14 10 90
15 90 20
16 30 80
DEMAND_SECTION
1 0
2 10
3 5
4 12
5 8
6 15
7 20
8 7
9 11
10 9
11 13
12 6
13 18
14 4
15 16
16 10
DEPOT_SECTION
1
-1
EOF
    """
    parser = TSPLibParser()
    parsed_data = parser.parse_string(file_content_cvrp)

    # Assert specification data
    assert parsed_data["specification"]["NAME"] == "P-n16-k8"
    assert parsed_data["specification"]["TYPE"] == "CVRP"
    assert parsed_data["specification"]["DIMENSION"] == 16
    assert parsed_data["specification"]["CAPACITY"] == 100
    assert parsed_data["specification"]["EDGE_WEIGHT_TYPE"] == "EUC_2D"

    # Assert node coordinates data (check a few key entries for brevity)
    assert "NODE_COORDS" in parsed_data["data"]
    assert len(parsed_data["data"]["NODE_COORDS"]) == 16
    assert math.isclose(parsed_data["data"]["NODE_COORDS"][1][0], 40.0)
    assert math.isclose(parsed_data["data"]["NODE_COORDS"][1][1], 50.0)
    assert math.isclose(parsed_data["data"]["NODE_COORDS"][16][0], 30.0)
    assert math.isclose(parsed_data["data"]["NODE_COORDS"][16][1], 80.0)

    expected_demands = {
        1: 0,
        2: 10,
        3: 5,
        4: 12,
        5: 8,
        6: 15,
        7: 20,
        8: 7,
        9: 11,
        10: 9,
        11: 13,
        12: 6,
        13: 18,
        14: 4,
        15: 16,
        16: 10,
    }
    assert "DEMANDS" in parsed_data["data"]
    assert parsed_data["data"]["DEMANDS"] == expected_demands

    # Assert depot data
    assert "DEPOT" in parsed_data["data"]
    assert parsed_data["data"]["DEPOT"] == [1]
