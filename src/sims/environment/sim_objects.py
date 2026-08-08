import json
from typing import Dict, Any

from torch.distributions.utils import lazy_property

from sims.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET,
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA,
)


# NOTE: Lazy import - don't import OBJAVERSE_ANNOTATIONS at module level
# Import it inside methods that need it to avoid network calls during import
def _get_objaverse_annotations():
    """Helper to lazily import OBJAVERSE_ANNOTATIONS only when needed."""
    from sims.utils.objaverse_utils import OBJAVERSE_ANNOTATIONS

    return OBJAVERSE_ANNOTATIONS


class SimObject(dict):
    ALWAYS_KEYS = {"isObjaverse", "synset", "lemma"}

    def __init__(self, thor_obj: Dict[str, Any]):
        super().__init__()
        self._thor_obj = thor_obj
        self._cache = {}

    @lazy_property
    def is_objaverse(self):
        return self._thor_obj["assetId"] in _get_objaverse_annotations()

    def __getitem__(self, item):
        if (
            self.is_objaverse
            and item == "objectType"
            and self._thor_obj[item] == "Undefined"
        ):
            return self._thor_obj["objectId"].split("|")[0]

        if item in self._thor_obj:
            return self._thor_obj[item]

        if item in self._cache:
            return self._cache[item]

        asset_id = self._thor_obj["assetId"]

        if item == "isObjaverse":
            return self.is_objaverse

        elif item == "synset":
            if self.is_objaverse:
                self._cache[item] = _get_objaverse_annotations()[asset_id]["synset"]
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET[
                    self._thor_obj["objectType"]
                ]

        elif item == "lemma":
            if self.is_objaverse:
                self._cache[item] = _get_objaverse_annotations()[asset_id][
                    "most_specific_lemma"
                ]
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA[
                    self._thor_obj["objectType"]
                ]

        elif self.is_objaverse and item in _get_objaverse_annotations()[asset_id]:
            self._cache[item] = _get_objaverse_annotations()[asset_id][item]

        elif not self.is_objaverse and item == "description":
            self._cache[item] = (
                f"undescribed THOR item, type {self._thor_obj['objectType']}"
            )

        else:
            raise ValueError(f"Unknown key {item}")

        return self._cache[item]

    def __setitem__(self, key, value):
        if key in self._thor_obj:
            self._thor_obj[key] = value
        else:
            self._cache[key] = value

    def _key_set(self):
        keys = set(self._thor_obj.keys())
        keys.update(self._cache.keys())
        keys.update(self.ALWAYS_KEYS)

        if self.is_objaverse:
            keys.update(_get_objaverse_annotations()[self._thor_obj["assetId"]].keys())

        return keys

    def keys(self):
        return iter(self._key_set())

    def values(self):
        return map(self.__getitem__, self.keys())

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __contains__(self, key):
        if key in self._thor_obj or key in self._cache or key in self.ALWAYS_KEYS:
            return True
        if not self.is_objaverse:
            return False
        return key in _get_objaverse_annotations()[self._thor_obj["assetId"]]

    def __eq__(self, other):
        if not isinstance(other, SimObject):
            return False
        return self._thor_obj == other._thor_obj and self._cache == other._cache

    def __str__(self):
        return json.dumps({**self._thor_obj, **self._cache}, indent=2)

    # noinspection PyStatementEffect
    def __repr__(self):
        return (
            f"SimObject("
            f"objectId={self['objectId']},"
            f" objectType={self['objectType']},"
            f" assetId={self['assetId']},"
            f" isObjaverse={self['isObjaverse']},"
            f")"
        )
