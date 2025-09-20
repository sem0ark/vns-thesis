import logging
import re
from typing import Any, Dict, List, Tuple, Union

from lark import Lark, Transformer, v_args

tsplib_grammar = r"""
    %import common.WS
    %ignore WS

    start: (specification_entry _NEWLINE)* (data_section _NEWLINE?)* _EOF_KEYWORD?

    specification_entry: KEYWORD ":" VALUE_STRING
    KEYWORD: /[A-Z_]+/
    VALUE_STRING: /[^\:\n\r]+/

    // Data Sections
    data_section: node_coord_section
                | depot_section
                | demand_section
                | edge_data_section
                | fixed_edges_section
                | display_data_section
                | tour_section
                | edge_weight_section

    node_coord_section: "NODE_COORD_SECTION" _NEWLINE (node_coord_line)+
    node_coord_line: INT (INT|REAL) (INT|REAL) (INT|REAL)? _NEWLINE

    depot_section: "DEPOT_SECTION" _NEWLINE (INT+ _NEWLINE)* _MINUS_ONE _NEWLINE

    demand_section: "DEMAND_SECTION" _NEWLINE (demand_line)+
    demand_line: INT INT _NEWLINE

    edge_data_section: "EDGE_DATA_SECTION" _NEWLINE (edge_list_line | adj_list_line)* _MINUS_ONE (_NEWLINE _MINUS_ONE)? _NEWLINE
    edge_list_line: INT INT _NEWLINE
    adj_list_line: INT (INT+)? _MINUS_ONE _NEWLINE

    fixed_edges_section: "FIXED_EDGES_SECTION" _NEWLINE (fixed_edge_line)+ _MINUS_ONE _NEWLINE
    fixed_edge_line: INT INT _NEWLINE

    display_data_section: "DISPLAY_DATA_SECTION" _NEWLINE (display_coord_line)+
    display_coord_line: INT REAL REAL _NEWLINE

    tour_section: "TOUR_SECTION" _NEWLINE (tour_line | _MINUS_ONE _NEWLINE)* _MINUS_ONE _NEWLINE
    tour_line: INT+ _NEWLINE

    edge_weight_section: "EDGE_WEIGHT_SECTION" _NEWLINE (edge_weight_line)+
    edge_weight_line: (INT|REAL)+ _NEWLINE

    INT: /-?\d+/
    REAL: /-?\d+\.\d+(e[+\-]?\d+)?/i
    _NEWLINE: /[\r?\n]+/
    _MINUS_ONE: "-1"
    _EOF_KEYWORD: "EOF"
"""


@v_args(inline=True)
class TSPLibTransformer(Transformer):
    def __init__(self):
        self.data: Dict[str, Any] = {"specification": {}, "data": {}}
        super().__init__()

    def start(self, *items):
        return self.data

    def specification_entry(self, keyword_token, value_string_token):
        keyword = str(keyword_token).strip()
        value = str(value_string_token).strip()

        if keyword in ["DIMENSION", "CAPACITY"]:
            self.data["specification"][keyword] = int(value)
        elif re.match(
            r"^-?\d+(\.\d+)?(e[+\-]?\d+)?$", value
        ):  # Check if it looks like a number
            try:
                self.data["specification"][keyword] = float(value)
            except ValueError:
                self.data["specification"][keyword] = (
                    value  # Keep as string if float conversion fails
                )
        else:
            self.data["specification"][keyword] = value
        return None  # Don't return anything to the parent, just update self.data

    def node_coord_section(self, *lines):
        coords = {}
        for line_data in lines:
            if len(line_data) == 3:  # 2D
                node_id, x, y = line_data
                coords[node_id] = (x, y)
            elif len(line_data) == 4:  # 3D
                node_id, x, y, z = line_data
                coords[node_id] = (x, y, z)
        self.data["data"]["NODE_COORDS"] = coords
        return None

    def node_coord_line(self, node_id, x, y, z=None):
        if z is not None:
            return int(node_id), float(x), float(y), float(z)
        return int(node_id), float(x), float(y)

    def depot_section(self, *ints):
        self.data["data"]["DEPOT"] = [int(x) for x in ints]
        return None

    def demand_section(self, *lines):
        demands = {}
        for line_data in lines:
            node_id, demand = line_data
            demands[node_id] = demand
        self.data["data"]["DEMANDS"] = demands
        return None

    def demand_line(self, node_id, demand):
        return int(node_id), int(demand)

    def edge_data_section(self, *lines):
        edge_data_format = self.data["specification"].get("EDGE_DATA_FORMAT")
        edges: Union[List[Tuple[int, int]], Dict[int, List[int]]] = []

        if edge_data_format == "ADJ_LIST":
            edges = {}
            for line_data in lines:
                if (
                    isinstance(line_data, tuple) and line_data[-1] == -1
                ):  # adj_list_line returns tuple (node_id, adj_nodes..., -1)
                    node_id = line_data[0]
                    adj_nodes = [int(n) for n in line_data[1:-1]]
                    edges[node_id] = adj_nodes
                # else: This might be the final -1 or empty. Handled by grammar.
        elif edge_data_format == "EDGE_LIST":
            for line_data in lines:
                if isinstance(
                    line_data, tuple
                ):  # edge_list_line returns tuple (node1, node2)
                    edges.append((int(line_data[0]), int(line_data[1])))

        self.data["data"]["EDGE_DATA"] = edges
        return None

    def edge_list_line(self, n1, n2):
        return int(n1), int(n2)

    def adj_list_line(self, node_id, *adj_nodes_and_minus_one):
        # adj_nodes_and_minus_one will be a list of tokens/values, ending with MINUS_ONE
        return int(node_id), *(int(n) for n in adj_nodes_and_minus_one)

    def fixed_edges_section(self, *lines):
        fixed_edges = []
        for line_data in lines:
            n1, n2 = line_data
            fixed_edges.append((n1, n2))
        self.data["data"]["FIXED_EDGES"] = fixed_edges
        return None

    def fixed_edge_line(self, n1, n2):
        return int(n1), int(n2)

    def display_data_section(self, *lines):
        display_coords = {}
        for line_data in lines:
            node_id, x, y = line_data
            display_coords[node_id] = (x, y)
        self.data["data"]["DISPLAY_DATA"] = display_coords
        return None

    def display_coord_line(self, node_id, x, y):
        return int(node_id), float(x), float(y)

    def tour_section(self, *tour_elements):
        tours = []
        current_tour = []
        for element in tour_elements:
            if isinstance(element, list):  # a tour_line (list of integers)
                current_tour.extend(element)
            elif str(element) == "-1":  # a MINUS_ONE token
                if current_tour:
                    tours.append(current_tour)
                    current_tour = []
        if current_tour:  # Add last tour if file ends without final -1 for it
            tours.append(current_tour)
        self.data["data"]["TOURS"] = tours
        return None

    def tour_line(self, *nodes):
        return [int(n) for n in nodes]  # Return list of integers for a single tour line

    def edge_weight_section(self, *lines):
        edge_weight_format = self.data["specification"].get("EDGE_WEIGHT_FORMAT")
        dimension = self.data["specification"].get("DIMENSION")

        all_weights = []
        for line_data in lines:
            all_weights.extend([float(x) for x in line_data])

        matrix: List[List[float]] = []
        if edge_weight_format == "FULL_MATRIX":
            if dimension:
                for i in range(dimension):
                    row = all_weights[i * dimension : (i + 1) * dimension]
                    matrix.append(row)
            else:
                logging.warning(
                    "DIMENSION not specified for EDGE_WEIGHT_SECTION (FULL_MATRIX). Storing as flat list."
                )
                matrix = all_weights

        elif edge_weight_format in [
            "UPPER_ROW",
            "LOWER_ROW",
            "UPPER_DIAG_ROW",
            "LOWER_DIAG_ROW",
        ]:
            matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
            weight_idx = 0
            for i in range(dimension):
                for j in range(dimension):
                    if (
                        (edge_weight_format == "UPPER_ROW" and i < j)
                        or (edge_weight_format == "LOWER_ROW" and i > j)
                        or (edge_weight_format == "UPPER_DIAG_ROW" and i <= j)
                        or (edge_weight_format == "LOWER_DIAG_ROW" and i >= j)
                    ):
                        if weight_idx < len(all_weights):
                            matrix[i][j] = all_weights[weight_idx]
                            weight_idx += 1

            # Fill symmetric parts for TSP/HCP if needed
            if self.data["specification"].get("TYPE") in ["TSP", "HCP"]:
                for i in range(dimension):
                    for j in range(i + 1, dimension):
                        if edge_weight_format in ["UPPER_ROW", "UPPER_DIAG_ROW"]:
                            matrix[j][i] = matrix[i][j]
                        elif edge_weight_format in ["LOWER_ROW", "LOWER_DIAG_ROW"]:
                            matrix[i][j] = matrix[j][i]
        else:
            raise ValueError(f"Cannot parse format of type {edge_weight_format}.")

        self.data["data"]["EDGE_WEIGHTS"] = matrix
        return None

    def edge_weight_line(self, *numbers):
        return [float(n) for n in numbers]

    def INT(self, token):
        return int(token)

    def REAL(self, token):
        return float(token)


class TSPLibParser:
    """
    Parses TSPLIB files using the Lark parsing library.
    """

    def __init__(self):
        # Create a Lark parser instance with the grammar
        # 'ignore_matches_in_production=True' is needed for %ignore NEWLINE in data sections
        self.parser = Lark(
            tsplib_grammar,
            start="start",
            propagate_positions=False,
            maybe_placeholders=False,
        )

    def parse_string(self, file_content: str) -> Dict[str, Any]:
        """
        Parses the TSPLIB file content string and returns a structured dictionary.
        """
        # Ensure proper newlines for Lark to parse correctly
        normalized_content = (
            file_content.replace("\r\n", "\n").strip() + "\n"
        )  # Ensure trailing newline

        # Parse the content into a Lark tree
        tree = self.parser.parse(normalized_content)

        # Transform the tree into the desired dictionary structure
        transformer = TSPLibTransformer()
        transformed_data = transformer.transform(tree)

        return transformed_data
