"""Minimal package marker for the standalone RootScope-Ω delta.

The production ``app/__init__.py`` intentionally exposes the complete
RootScope control API and therefore imports modules that are outside the Ω
delta.  The release builder maps this file to ``rootscope/app/__init__.py`` so
an extracted delta can import the Ω packages without importing any production
runtime, hardware, serial, or state-machine module.
"""

__all__: list[str] = []
