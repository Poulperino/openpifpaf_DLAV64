BBOX_SKELETON_COMPLETE = [
    [1, 2], [2, 3], [3, 4], [4, 1], [1,5], [2,5], [3,5], [4,5]
]

BBOX_KEYPOINTS = [
    'top_left',
    'top_right',
    'bottom_right',
    'bottom_left',
    'center',
]

HFLIP = {
    'top_left': 'top_right',
    'top_right': 'top_left',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
}

VISDRONE_CATEGORIES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    #"others"
]

VISDRONE_BBOX_KEYPOINTS = []
VISDRONE_HFLIP = {}
for categ in VISDRONE_CATEGORIES:
    for loc in BBOX_KEYPOINTS:
        VISDRONE_BBOX_KEYPOINTS.append(loc + '_' + categ)
    for source, target in HFLIP.items():
        VISDRONE_HFLIP[source + '_' + categ] = target + '_' + categ

# VISDRONE_BBOX_KEYPOINTS = [
#     'top_left_pedestrian',
#     'top_right_pedestrian',
#     'bottom_right_pedestrian',
#     'bottom_left_pedestrian',
#     'center_pedestrian',
# ]

UAVDT_KEYPOINTS = [
    'center',
]

UAVDT_KEYPOINTS_3CATEGORIES = [
    'center_car',
    'center_truck',
    'center_bus',
]

UAVDT_CATEGORIES = [
    "car",
    "truck",
    "bus",
]


UAVDT_BBOX_3CATEGORIES = []
UAVDT_HFLIP_3CATEGORIES = {}
for categ in UAVDT_CATEGORIES:
    for loc in BBOX_KEYPOINTS:
        UAVDT_BBOX_3CATEGORIES.append(loc + '_' + categ)
    for source, target in HFLIP.items():
        UAVDT_HFLIP_3CATEGORIES[source + '_' + categ] = target + '_' + categ

VISDRONE_KEYPOINTS = [
    "center_pedestrian",
    "center_people",
    "center_bicycle",
    "center_car",
    "center_van",
    "center_truck",
    "center_tricycle",
    "center_awning-tricycle",
    "center_bus",
    "center_motor",
    #"others"
]
