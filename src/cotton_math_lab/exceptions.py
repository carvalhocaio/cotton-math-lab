"""Hierarquia de exceções do laboratório."""


class CottonMathLabError(Exception):
    """Base de todas as exceções do pacote."""


class InvalidSpecError(CottonMathLabError, ValueError):
    """Parâmetros de uma especificação são internamente inconsistentes."""


class LinAlgError(CottonMathLabError):
    """Falha em operação de álgebra linear."""
