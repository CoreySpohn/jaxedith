"""Shared fixtures for jaxedith tests."""

import os

from jaxtyping import install_import_hook

# Runtime shape and dtype checking of the jaxtyping annotations, tests only.
# The hook must be installed before jaxedith is first imported. Set
# JAXEDITH_TYPECHECK=0 to run without it.
if os.environ.get("JAXEDITH_TYPECHECK", "1") == "1":
    install_import_hook("jaxedith", "beartype.beartype")
