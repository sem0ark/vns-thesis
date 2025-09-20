"""
Implements various distance calculation methods as specified in TSPLIB format.
Distances are rounded to the nearest integer as per TSPLIB specification.
"""

import math
from typing import Any, Callable


def _nint(x: float) -> int:
    """
    Rounds a float to the nearest integer. Corresponds to C's nint(x).
    """
    return int(x + 0.5)


def euc_2d(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    2.1 Euclidean distance (L2-metric)

    Calculates Euclidean distance for 2D coordinates.
    dij = nint( sqrt( xd*xd + yd*yd) );
    """
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    return _nint(math.sqrt(xd * xd + yd * yd))


def euc_3d(
    coord1: tuple[float, float, float], coord2: tuple[float, float, float]
) -> int:
    """
    Calculates Euclidean distance for 3D coordinates.
    dij = nint( sqrt( xd*xd + yd*yd + zd*zd) );
    """
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    zd = coord1[2] - coord2[2]
    return _nint(math.sqrt(xd * xd + yd * yd + zd * zd))


def man_2d(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    2.2 Manhattan distance (L1-metric)

    Calculates Manhattan distance for 2D coordinates.
    dij = nint( abs(xd) + abs(yd) );
    """
    xd = abs(coord1[0] - coord2[0])
    yd = abs(coord1[1] - coord2[1])
    return _nint(xd + yd)


def man_3d(
    coord1: tuple[float, float, float], coord2: tuple[float, float, float]
) -> int:
    """
    Calculates Manhattan distance for 3D coordinates.
    dij = nint( abs(xd) + abs(yd) + abs(zd) );
    """
    xd = abs(coord1[0] - coord2[0])
    yd = abs(coord1[1] - coord2[1])
    zd = abs(coord1[2] - coord2[2])
    return _nint(xd + yd + zd)


def max_2d(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    2.3 Maximum distance (L∞-metric)

    Calculates Maximum distance for 2D coordinates.
    dij = max( nint(abs(xd)), nint(abs(yd)) );
    """
    xd = abs(coord1[0] - coord2[0])
    yd = abs(coord1[1] - coord2[1])
    return max(_nint(xd), _nint(yd))


def max_3d(
    coord1: tuple[float, float, float], coord2: tuple[float, float, float]
) -> int:
    """
    Calculates Maximum distance for 3D coordinates.
    dij = max( nint(abs(xd)), nint(abs(yd)), nint(abs(zd)) );
    """
    xd = abs(coord1[0] - coord2[0])
    yd = abs(coord1[1] - coord2[1])
    zd = abs(coord1[2] - coord2[2])
    return max(_nint(xd), _nint(yd), _nint(zd))


def _to_radians_geo(coord_val: float) -> float:
    """Converts DDD.MM geographical format to r
    2.4 Geographical distance in radians."""
    PI = 3.141592
    deg = _nint(coord_val)
    _min = coord_val - deg
    return PI * (deg + 5.0 * _min / 3.0) / 180.0


def geo(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    Calculates geographical distance between two points on the idealized sphere.
    Coordinates are expected in DDD.MM format.
    """
    RRR = 6378.388  # Radius of the idealized sphere in km

    lat1 = _to_radians_geo(coord1[0])
    lon1 = _to_radians_geo(coord1[1])
    lat2 = _to_radians_geo(coord2[0])
    lon2 = _to_radians_geo(coord2[1])

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)

    # Ensure argument to acos is within [-1, 1] due to floating point inaccuracies
    acos_arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    acos_arg = max(-1.0, min(1.0, acos_arg))  # Clamp to valid range

    dij = RRR * math.acos(acos_arg)
    return int(dij + 1.0)  # Explicitly (int) (dij + 1.0) as per C implementation


def att(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    2.5 Pseudo-Euclidean distance (ATT)

    Calculates special "pseudo-Euclidean" distance for ATT problems.
    """
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]

    rij = math.sqrt((xd * xd + yd * yd) / 10.0)
    tij = _nint(rij)

    if tij < rij:
        dij = tij + 1
    else:
        dij = tij
    return dij


def ceil_2d(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """
    2.6 Ceiling of the Euclidean distance (CEIL_2D)

    Calculates 2D Euclidean distance and rounds it up to the next integer.
    """
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    return math.ceil(math.sqrt(xd * xd + yd * yd))


def xray1(
    coord1: tuple[float, float, float], coord2: tuple[float, float, float]
) -> int:
    """
    2.7 Distance for crystallography problems (XRAY1, XRAY2)

    Placeholder for XRAY1 distance function.
    The specification points to external FORTRAN implementations (deq.f).
    This would require reimplementing the PHI, CHI, TWOTH functions and the specific logic.
    For demonstration, it returns a symbolic value or raises NotImplementedError.
    """
    # PHI, CHI, TWOTH are the respective x, y, z coordinates in this context
    phi1, chi1, twoth1 = coord1
    phi2, chi2, twoth2 = coord2

    distp = min(abs(phi1 - phi2), abs(abs(phi1 - phi2) - 360.0))
    distc = abs(chi1 - chi2)
    distt = abs(twoth1 - twoth2)

    cost = max(distp / 1.00, distc / 1.0, distt / 1.00)

    # Multiply by 100.0 and round to the nearest integer as proposed
    return _nint(100.0 * cost)


# Mapping of EDGE_WEIGHT_TYPE strings to their corresponding distance functions
_DISTANCE_FUNCTION_MAP = {
    "EUC_2D": euc_2d,
    "EUC_3D": euc_3d,
    "MAN_2D": man_2d,
    "MAN_3D": man_3d,
    "MAX_2D": max_2d,
    "MAX_3D": max_3d,
    "GEO": geo,
    "ATT": att,
    "CEIL_2D": ceil_2d,
    "XRAY1": xray1,
}


def get_distance_function(edge_weight_type: str) -> Callable[[Any, Any], int]:
    """
    Returns the appropriate distance calculation function based on the EDGE_WEIGHT_TYPE string.
    Raises ValueError if the type is unknown or not explicitly handled by a function.
    """
    func = _DISTANCE_FUNCTION_MAP.get(edge_weight_type.upper())
    if func:
        return func
    elif edge_weight_type.upper() in ["EXPLICIT", "SPECIAL"]:
        raise ValueError(
            f"Distance type '{edge_weight_type}' is handled via explicit data or special external logic, not a calculated function."
        )
    else:
        raise ValueError(
            f"Unknown or unsupported EDGE_WEIGHT_TYPE: '{edge_weight_type}'"
        )
