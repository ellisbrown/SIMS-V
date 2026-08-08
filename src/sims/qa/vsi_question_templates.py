# VSI-Bench question wording (official Apache-2.0 repository):
# https://github.com/vision-x-nyu/thinking-in-space

# VSI_BENCH
PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."
NA_POST_PROMPT_NUM = "Do not respond anything other than a single number!"


# OBJ_COUNTING_TEMPLATE = """
# These are frames of a video. How many {category}(s) are in this room? Please answer me with a single number.
# """.strip()

# This VSI-Bench wording is part of the released paper data. Labels count
# distinct trajectory-salient instances; see ``preprocess_salient_objects``.
OBJ_COUNTING_TEMPLATE = """
How many {category}(s) are in this room?
""".strip()

# OBJ_SIZE_ESTIMATE_TEMPLATE = """
# These are frames of a video. What is the length of the {category} in this room (length refers to the longest horizontal edge)?
# Options: A. {option_a} cm, B. {option_b} cm, C. {option_c} cm, D. {option_d} cm.
# Please directly answer A, B, C, D.
# """.strip()

# OBJ_SIZE_ESTIMATE_TEMPLATE = """
# What is the length of the {category} in this room (in cm and where length refers to the longest horizontal edge)?
# """.strip()

OBJ_SIZE_ESTIMATE_TEMPLATE = """
What is the length of the longest dimension (length, width, or height) of the {category}, measured in centimeters?""".strip()

# ROOM_SIZE_TEMPLATE = """
# These are frames of a video. What is the size of this room (in square meters)?
# Options: A. {choice_a} sq m, B. {choice_b} sq m, C. {choice_c} sq m, D. {choice_d} sq m.
# Please directly answer A, B, C, D.
# """.strip()

ROOM_SIZE_TEMPLATE = """
What is the size of this room (in square meters)?
If multiple rooms are shown, estimate the size of the combined space.
""".strip()

# OBJ_ABS_DISTANCE_TEMPLATE = """
# These are frames of a video. Measuring from the closest point of each object, what is the distance between {object1} and {object2} (in meters)?
# Options. A. {choice_a} m, B. {choice_b} m, C. {choice_c} m, D. {choice_d} m.
# Please directly answer A, B, C, D.
# """.strip()

OBJ_ABS_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, what is the direct distance between the {object1} and the {object2} (in meters)?
If there are multiple instances of an object category, use the pair of instances with the shortest distance.
""".strip()

# OBJ_REL_DISTANCE_TEMPLATE = """
# These are frames of a video. Measuring from the closest point of each object, which of these objects is the closest to the {category}?
# Options: A. {choice_a}, B. {choice_b}, C. {choice_c}, D. {choice_d}.
# Please directly answer A, B, C, D.
# """.strip()

OBJ_REL_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the {category}?
If there are multiple instances of an object category, measure to the closest.
""".strip()

# OBJ_REL_DIRECTION_TEMPLATE = """
# These are frames of a video.
# If I am standing in the center of the room and facing the {orienting_object}, what direction is the {querying_object} with respect to me?
# Options: A. front, B. back, C. left, D. right. Each direction covers 90 degrees, with "front" referring to the [-45, 45] degree range in front of me, and so on.
# Please directly answer A, B, C, D.
# """.strip()

OBJ_REL_DIRECTION_HARD_TEMPLATE = OBJ_REL_DIRECTION_V1_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?
The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).
""".strip()

# OBJ_REL_DIRECTION_V2_TEMPLATE = """
# If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?
# An object is to my back if I would have to turn at least 135 degrees in order to face it.
# Options: A. left, B. right, C. back.
# Please directly answer A, B, C.
# """.strip()

OBJ_REL_DIRECTION_MEDIUM_TEMPLATE = OBJ_REL_DIRECTION_V2_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?
An object is to my back if I would have to turn at least 135 degrees in order to face it.
""".strip()

# OBJ_REL_DIRECTION_V3_TEMPLATE = """
# These are frames of a video.
# If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
# Options: A. left, B. right.
# Please directly answer A or B.
# """.strip()

OBJ_REL_DIRECTION_EASY_TEMPLATE = OBJ_REL_DIRECTION_V3_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
""".strip()


OBJ_SPTP_DISTANCE_TEMPLATE = """
Which of the these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the ego-position at the last frame in the video?
""".strip()


OBJ_APPEARANCE_ORDER_TEMPLATE = """
What will be the first-time appearance order of the following categories in the video: {choice_a}, {choice_b}, {choice_c}, {choice_d}?
""".strip()


# NOT IN THEIR TEMPLATES
ROUTE_PLANNING_TEMPLATE = """
You are a robot beginning at the {start_obj} and facing the {orienting_obj}. You want to navigate to the {end_obj}. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): {actions} You have reached the final destination.
""".strip()
