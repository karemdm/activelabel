"""Standalone training scripts, run by the runner with the Python of the training environment.

They never import ``activelabel`` (it is not installed there): only the standard library and the
training packages. Each reads a ``spec.json`` and prints one JSON object per line.
"""
