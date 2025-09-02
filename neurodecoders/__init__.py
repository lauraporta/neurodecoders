from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neurodecoders")
except PackageNotFoundError:
    # package is not installed
    pass

__all__: list[str] = []
