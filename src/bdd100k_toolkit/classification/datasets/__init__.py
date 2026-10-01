"""Classification dataset registry: importing this package registers every adapter."""

from bdd100k_toolkit.classification import registry
from bdd100k_toolkit.classification.datasets import period, scenario, weather  # noqa: F401

get = registry.get
get_spec = registry.get_spec
list_datasets = registry.list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
