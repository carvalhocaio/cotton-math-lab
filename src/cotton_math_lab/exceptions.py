"""Exception hierarchy for the lab."""


class CottonMathLabError(Exception):
    """Base for all exceptions in the package."""


class InvalidSpecError(CottonMathLabError, ValueError):
    """Parameters of a spec are internally inconsistent."""


class LinAlgError(CottonMathLabError):
    """Failure in a linear algebra operation."""


class AutodiffError(CottonMathLabError):
    """Invalid use of the automatic differentiation graph."""
