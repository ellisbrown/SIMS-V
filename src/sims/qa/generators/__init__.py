from sims.qa.generators.common import (
    ALL_OBJECTS,
    ALL_OBJECTS_CLEAN,
    MIN_NUM_FRAMES,
    VSI_OBJECTS,
    VSI_OBJECTS_CLEAN,
    clean_obj_type,
    gen_count_options,
    gen_distance_options,
    gen_mc_question,
    generate_qa_batch,
)
from sims.qa.generators.descriptive import (
    DESCRIPTIVE_QA_GEN_FNS,
    gen_binary_qa,
    gen_binary_qas,
    gen_obj_count_qas,
    gen_obj_size_est_qas,
)
from sims.qa.generators.layout import (
    LAYOUT_QA_GEN_FNS,
    gen_house_size_est_qas,
    gen_n_rooms_qas,
)
from sims.qa.generators.spatial import (
    SPATIAL_QA_GEN_FNS,
    calc_3d_bbox_distance_between_objects,
    calculate_relative_direction,
    gen_obj_abs_dist_qa,
    gen_obj_abs_dist_qas,
    gen_obj_rel_dir_qa,
    gen_obj_rel_dir_qas,
    gen_obj_rel_dist_qa,
    gen_obj_rel_dist_qas,
)
from sims.qa.generators.temporal import (
    TEMPORAL_QA_GEN_FNS,
    gen_temporal_order_qa,
    gen_temporal_order_qas,
    gen_temporal_rel_qa,
    gen_temporal_rel_qas,
)

# Assemble unified QA_GEN_FNS from all submodules
_all_dicts = [
    TEMPORAL_QA_GEN_FNS,
    DESCRIPTIVE_QA_GEN_FNS,
    SPATIAL_QA_GEN_FNS,
    LAYOUT_QA_GEN_FNS,
]

QA_GEN_FNS = {}
for _d in _all_dicts:
    _overlap = set(QA_GEN_FNS) & set(_d)
    if _overlap:
        raise RuntimeError(f"Duplicate QA_GEN_FNS keys: {_overlap}")
    QA_GEN_FNS.update(_d)

__all__ = [
    # Unified registry
    "QA_GEN_FNS",
    # Per-module registries
    "TEMPORAL_QA_GEN_FNS",
    "DESCRIPTIVE_QA_GEN_FNS",
    "SPATIAL_QA_GEN_FNS",
    "LAYOUT_QA_GEN_FNS",
    # Constants
    "ALL_OBJECTS",
    "ALL_OBJECTS_CLEAN",
    "VSI_OBJECTS",
    "VSI_OBJECTS_CLEAN",
    "MIN_NUM_FRAMES",
    # Common helpers
    "clean_obj_type",
    "gen_mc_question",
    "generate_qa_batch",
    "gen_count_options",
    "gen_distance_options",
    # Temporal generators
    "gen_temporal_rel_qa",
    "gen_temporal_rel_qas",
    "gen_temporal_order_qa",
    "gen_temporal_order_qas",
    # Descriptive generators
    "gen_binary_qa",
    "gen_binary_qas",
    "gen_obj_count_qas",
    "gen_obj_size_est_qas",
    # Spatial generators
    "calc_3d_bbox_distance_between_objects",
    "calculate_relative_direction",
    "gen_obj_abs_dist_qa",
    "gen_obj_abs_dist_qas",
    "gen_obj_rel_dist_qa",
    "gen_obj_rel_dist_qas",
    "gen_obj_rel_dir_qa",
    "gen_obj_rel_dir_qas",
    # Layout generators
    "gen_n_rooms_qas",
    "gen_house_size_est_qas",
]
