"""Shared IIIF manifest URI rewriting helpers.

Gallica manifests are downloaded through the openapi.bnf.fr endpoint, but the
source CSVs reference the gallica.bnf.fr form (sometimes over plain http).
Every rewrite in the pipeline must go through these helpers so the forms stay
interchangeable when re-identifying a source at any stage.
"""
from typing import List

GALLICA_PREFIX = "https://gallica.bnf.fr/iiif/ark:/12148/"
GALLICA_PREFIX_HTTP = "http://gallica.bnf.fr/iiif/ark:/12148/"
OPENAPI_PREFIX = "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/"


def to_openapi(uri: str) -> str:
    return uri.replace(GALLICA_PREFIX, OPENAPI_PREFIX).replace(GALLICA_PREFIX_HTTP, OPENAPI_PREFIX)


def to_gallica(uri: str) -> str:
    return uri.replace(OPENAPI_PREFIX, GALLICA_PREFIX).replace(GALLICA_PREFIX_HTTP, GALLICA_PREFIX)


def uri_variants(uri: str) -> List[str]:
    """All known equivalent forms of a manifest URI, original form first."""
    return list(dict.fromkeys([
        uri,
        to_gallica(uri),
        to_openapi(uri),
        to_gallica(uri).replace(GALLICA_PREFIX, GALLICA_PREFIX_HTTP),
    ]))
