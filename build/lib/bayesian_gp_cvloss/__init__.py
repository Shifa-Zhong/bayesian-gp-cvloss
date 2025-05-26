# This file makes Python treat the directory as a package.
# It can also be used to expose parts of the package at the top level.

from .optimizer import GPCrossValidatedOptimizer

__all__ = ['GPCrossValidatedOptimizer'] 