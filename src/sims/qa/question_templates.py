# >>>>> DESCRIPTIVE / RECOGNITION <<<<< #
BINARY_RECOG_TEMPLATE = """
Did you see a '{type}' anywhere in the house?
""".strip()

# This wording is part of the released paper data. Labels count distinct
# trajectory-salient instances; see ``preprocess_salient_objects``.
OBJ_COUNTING_TEMPLATE = """
How many total {type}(s) are in this house?
""".strip()

# >>>>> TEMPORAL <<<<< #

TEMPORAL_RELATION_TEMPLATE = """
Did you see the {type_1} or the {type_2} first?
""".strip()

TEMPORAL_APPEARANCE_ORDER_TEMPLATE = """
What is the order in which the following objects appear for the first time in the video: {objects}?
""".strip()

# >>>>> SPATIAL <<<<< #

OBJ_ABS_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, what is the distance between the {ref_obj} and the {other_obj} (in meters)?
If there are multiple instances of an object category, use the pair of instances with the shortest distance.
""".strip()

OBJ_REL_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({a}, {b}, {c}, {d}) is the closest to the {type}?
If there are multiple instances of an object category, measure to the closest.
""".strip()

OBJ_REL_DIRECTION_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
""".strip()

OBJ_SIZE_ESTIMATION_TEMPLATE = """
What is the length of the {long_short} dimension (length, width, or height) of the {obj} (in centimeters)?
If there were multiple instances of the object, use the first one you saw.
""".strip()

OBJ_SIZE_ESTIMATION_TEMPLATE_V2 = """
Approximately how many centimeters is the {long_short} dimension (length/width/height) of the {obj}?
If there were multiple instances of the object, use the first one you saw.
""".strip()

HOUSE_SIZE_ESTIMATION_TEMPLATE = """
Approximately how many square meters is the entire house you toured, including all {n} rooms?
""".strip()

N_ROOMS_TEMPLATE = """
How many total rooms are in this house?
""".strip()

# TODO: need to determine final agent position
FINAL_ROOM_SIZE_ESTIMATION_TEMPLATE = """
Approximately how many square meters is the final room you entered?
""".strip()
