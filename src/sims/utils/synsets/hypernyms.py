from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Union, Sequence, Set, Dict, Iterable

from nltk.corpus import wordnet2022 as wn
from nltk.corpus.reader import Synset

from sims.utils.constants.object_constants import AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET
from sims.utils.objaverse_utils import get_objaverse_annotations

# Excluded hypernyms were created by running get_hypernym_to_descendants_for_all_known_synsets()
# and grabbing ally hypernyms that had >= 10 descendants. I then manually removed those that
# were too general or otherwise not useful.
EXCLUDED_HYPERNYMS = frozenset(
    {
        "entity.n.01",
        "physical_entity.n.01",
        "object.n.01",
        "whole.n.02",
        "artifact.n.01",
        "instrumentality.n.03",
        "device.n.01",
        "abstraction.n.06",
        "matter.n.03",
        "implement.n.01",
        "commodity.n.01",
        "consumer_goods.n.01",
        "solid.n.01",
        "substance.n.07",
        "living_thing.n.01",
        "relation.n.01",
        "part.n.01",
        "substance.n.01",
        "communication.n.02",
        "structure.n.01",
        "creation.n.02",
        "material.n.01",
        "attribute.n.02",
        "reproductive_structure.n.01",
        "shape.n.02",
        "article.n.02",
        "mechanism.n.05",
        "psychological_feature.n.01",
        "ware.n.01",
        "durables.n.01",
        "sheet.n.06",
        "measure.n.02",
        "part.n.03",
        "body_part.n.01",
        "signal.n.01",
        "symbol.n.01",
        "area.n.05",
        "representation.n.02",
        "framework.n.03",
        "supporting_structure.n.01",
        "part.n.02",
        "causal_agent.n.01",
        "act.n.02",
        "event.n.01",
        "natural_object.n.01",
        "work.n.02",
        "production.n.02",
        "system_of_measurement.n.01",
        "plant_part.n.01",
        "plant_organ.n.01",
        "substance.n.08",
        "cognition.n.01",
        "solid.n.03",
        "matter.n.01",
        "thing.n.12",
        "matter.n.02",
        "group.n.01",
        "chordate.n.01",
        "covering.n.02",
        "organism.n.01",
        "transducer.n.01",
        "representational_process.n.01",
        "tube.n.01",
        "conduit.n.01",
        "way.n.06",
        "passage.n.03",
        "collection.n.01",
        "padding.n.01",
        "product.n.02",
        "rotating_mechanism.n.01",
    }
)


def generate_all_hypernyms_with_exclusions(
    synset: Union[str, Synset],
    excluded: Union[Set[str], str] = EXCLUDED_HYPERNYMS,
    include_self_synset: bool = True,
) -> Set[Synset]:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    return set(
        h
        for hp in synset.hypernym_paths()
        for h in hp
        if (include_self_synset or h != synset) and h.name() not in excluded
    )


@lru_cache(maxsize=10000, typed=True)
def is_hypernym_of(
    synset: Union[str, Synset], possible_hypernym: Union[str, Synset]
) -> bool:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    if isinstance(possible_hypernym, str):
        possible_hypernym = wn.synset(possible_hypernym)

    return possible_hypernym in synset.lowest_common_hypernyms(possible_hypernym)


def is_subsynset_of(
    synset: Union[str, Synset], other_synset: Union[str, Synset]
) -> bool:
    return is_hypernym_of(synset=synset, possible_hypernym=other_synset)


def symmetric_subsynset_of(
    synset: Union[str, Synset], other_synset: Union[str, Synset]
) -> bool:
    return is_hypernym_of(
        synset=synset, possible_hypernym=other_synset
    ) or is_hypernym_of(synset=other_synset, possible_hypernym=synset)


def generate_hypernym_to_descendants(
    synsets: Union[Sequence[str], Sequence[Synset]],
) -> Dict[str, List[Synset]]:
    if len(synsets) == 0:
        return {}

    if isinstance(synsets[0], str):
        synsets = [wn.synset(s) for s in synsets]

    synsets = set(synsets)
    synsets = [s.name() for s in synsets]

    hypernym_to_descendants = defaultdict(list)
    for s in synsets:
        s = wn.synset(s)
        paths = s.hypernym_paths()
        for hypernym in set(sum(paths, [])):
            hypernym_to_descendants[hypernym.name()].append(s)

    return hypernym_to_descendants


def filter_synsets_to_remove_hyponyms(
    synsets: Union[Sequence[str], Sequence[Synset]],
) -> List[str]:
    if len(synsets) == 0:
        return []

    hyper_to_descs = generate_hypernym_to_descendants(synsets=synsets)

    if isinstance(synsets[0], Synset):
        synsets = [s.name() for s in synsets]

    to_remove = set()
    for synset in synsets:
        descs = hyper_to_descs[synset]
        if len(descs) > 1:
            for desc in descs:
                if desc.name() != synset:
                    to_remove.add(desc.name())

    return list(set(synsets) - to_remove)


def get_all_known_synsets() -> List[Synset]:
    anns = get_objaverse_annotations()
    synsets = set(ann["synset"] for ann in anns.values()) | set(
        AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()
    )
    synsets = sorted(list(set([wn.synset(s) for s in synsets])), key=lambda s: s.name())
    return synsets


def get_hypernym_to_descendants_for_all_known_synsets():
    synsets = get_all_known_synsets()
    return generate_hypernym_to_descendants(synsets)


@lru_cache(maxsize=10000, typed=True)
def get_hyponyms_of_synset(
    synset: Union[str, Synset], return_strings: bool
) -> Union[Set[Synset], Set[str]]:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    if return_strings:
        hyps = {synset.name()}
    else:
        hyps = {synset}

    for h in synset.hyponyms():
        hyps.update(
            iter(
                get_hyponyms_of_synset(
                    h,
                    return_strings=return_strings,
                )
            )
        )

    return hyps


def get_hyponyms_of_synsets(
    synsets: Union[Iterable[str], Iterable[Synset]], return_strings: bool
) -> Union[Set[Synset], Set[str]]:
    hyponyms: Union[Set[Synset], Set[str]] = set()
    for s in synsets:
        hyponyms.update(iter(get_hyponyms_of_synset(s, return_strings=return_strings)))

    return hyponyms


@lru_cache(maxsize=None)
def get_singleton_highest_hypernyms():
    highest_hypernyms = Counter()
    for syn in (
        list(AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()) + get_all_known_synsets()
    ):
        highest_hypernyms[get_highest_relevant_hypernym(syn)] += 1

    return set([h for h in highest_hypernyms if highest_hypernyms[h] < 2])


def get_highest_relevant_hypernym(
    synset: Union[str, Synset],
    excluded: Union[Set[str], str] = EXCLUDED_HYPERNYMS,
) -> Synset:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    for hpath in synset.hypernym_paths():
        for hyp in hpath:
            if hyp.name() not in excluded:
                return hyp.name()

    return synset.name()  # return self if no non-excluded hypernyms


# assert __name__ == "__main__"
#
# anns = get_objaverse_annotations()
#
# synsets = set(ann["synset"] for ann in anns.values()) | set(
#     AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()
# )
# synsets = set([wn.synset(s) for s in synsets])
# synsets = [s.name() for s in synsets]
#
#
# hypernym_to_descendants = defaultdict(list)
# for s in synsets:
#     s = wn.synset(s)
#     paths = s.hypernym_paths()
#     for hypernym in set(sum(paths, [])):
#         hypernym_to_descendants[hypernym.name()].append(s)
#
# hypernym_to_count = Counter({k: len(v) for k, v in hypernym_to_descendants.items()})
