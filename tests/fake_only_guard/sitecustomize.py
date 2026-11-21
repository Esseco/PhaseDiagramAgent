"""Opt-in subprocess guard against known real scientific model loaders.

Prepend this directory to PYTHONPATH only for synthetic regression runs.
Python -I/-S ignores sitecustomize; those tests must provide their own fakes.
"""

import sys
from importlib.abc import MetaPathFinder


class _ModelLoaderGuard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "maml.describers" or fullname.startswith(
            (
                "chgnet.model",
                "mace.calculators",
                "m3gnet.models",
                "sevenn.calculator",
                "orb_models.forcefield",
                "matgl.ext.ase",
            )
        ):
            raise RuntimeError("real_model_loader_blocked:" + fullname)


sys.meta_path.insert(0, _ModelLoaderGuard())
