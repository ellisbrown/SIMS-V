import random
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Set, Sequence, List, Tuple, Iterable, Literal, Union

import torch
import numpy as np
from ai2thor.controller import Controller
from ai2thor.server import Event
from shapely import Polygon, GeometryCollection, Point

from sims.environment.action_spaces import agent_alignment_to_point
from sims.environment.actions import (
    StretchAction,
    StretchGraspAction,
    StretchDropOffAction,
)
from sims.environment.stretch_state import StretchState
from sims.environment.sim_objects import SimObject
from sims.utils.constants.stretch_initialization_utils import (
    AGENTS_BASE_HEIGHT,
    INTEL_VERTICAL_FOV,
    AGENT_RADIUS_LIST,
    GRID_SIZE,
    ADDITIONAL_ARM_ARGS,
    HORIZON,
    ADDITIONAL_NAVIGATION_ARGS,
    STRETCH_COMMIT_ID,
    STRETCH_CAMERA_HORIZONTAL_CROP,
)
from sims.utils.data_generation_utils.navigation_utils import (
    get_rooms_polymap_and_type,
    get_room_id_from_location,
    get_wall_center_floor_level,
    triangulate_room_polygon,
    is_any_object_sufficiently_visible_and_in_center_frame,
    snap_to_skeleton,
)

from sims.utils.distance_calculation_utils import sum_dist_path, position_dist
from sims.utils.synsets.hypernyms import is_hypernym_of
from sims.utils.type_utils import Vector3


class StretchController:
    def __init__(
        self,
        initialize_controller=True,
        rotation_noise_std_degrees=0.0,
        **kwargs,
    ):
        self.rotation_noise_std_degrees = rotation_noise_std_degrees
        self.should_render_image_synthesis = (
            kwargs.get("renderDepthImage", False)
            or kwargs.get("renderNormalsImage", False)
            or kwargs.get("renderFlowImage", False)
            or kwargs.get("renderObjectImage", False)
            or kwargs.get("renderClassImage", False)
            or kwargs.get("renderInstanceSegmentation", False)
            or kwargs.get("renderSemanticSegmentation", False)
        )
        self.mode = None

        self.room_poly_map: Optional[Dict[str, Polygon]] = None
        self.room_type_dict: Optional[Dict[str, str]] = None

        # The usage here is to judge if a spatial action counts as successful or not
        self._universal_state_tolerance = StretchState._create_difference_state(
            diff_base={"x": 0.01, "z": 0.01, "theta": 1.5},
            diff_wrist={"y": 0.005, "z": 0.005, "yaw": 2},
            diff_hand={
                "x": 100,
                "y": 100,
                "z": 100,
            },  # direct hand is a no-op
            diff_gripper=100,
            diff_held_oids=set(),
        )

        self._current_horizon = 0
        self._nav_visible_objects_cache = {}
        self._manip_visible_objects_cache = {}

        if initialize_controller:
            self.controller = Controller(**kwargs)
            self.initialization_args = kwargs
            print(f"Using Controller commit id: {self.controller._build.commit_id}")
            assert STRETCH_COMMIT_ID in self.controller._build.commit_id

            if "scene" in kwargs:
                self.reset(kwargs["scene"])

            def is_fov_correct():
                third_party_cams = self.controller.last_event.metadata.get(
                    "thirdPartyCameras"
                )
                if not third_party_cams:
                    return (
                        True  # No scene loaded yet; FOV will be checked after reset()
                    )
                return abs(third_party_cams[0]["fieldOfView"] - INTEL_VERTICAL_FOV) < 2

            if not is_fov_correct():
                self.controller.step(
                    "UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    fieldOfView=INTEL_VERTICAL_FOV,
                )
                assert is_fov_correct()

            self.camera_width = self.controller.width
            print(f"Camera width: {self.camera_width}")

    def get_controller_camera_params(self, which_camera: Literal["nav", "manip"]):
        if which_camera == "nav":
            camera_rel_position = self.controller.last_event.metadata[
                "agentPositionRelativeCameraPosition"
            ]
            camera_rel_rotation = self.controller.last_event.metadata[
                "agentPositionRelativeCameraRotation"
            ]
            fov_y = self.controller.last_event.metadata["fov"]
        elif which_camera == "manip":
            camera_rel_position = self.controller.last_event.metadata[
                "thirdPartyCameras"
            ][0]["agentPositionRelativeThirdPartyCameraPosition"]
            camera_rel_rotation = self.controller.last_event.metadata[
                "thirdPartyCameras"
            ][0]["agentPositionRelativeThirdPartyCameraRotation"]
            fov_y = self.controller.last_event.metadata["thirdPartyCameras"][0][
                "fieldOfView"
            ]
        else:
            raise ValueError(f"Invalid camera: {which_camera}")

        return camera_rel_position, camera_rel_rotation, fov_y

    def get_object_from_pixel(
        self, point: List[int], which_camera: Literal["nav", "manip"]
    ):
        if which_camera == "nav":
            frame = self.navigation_camera
            masks_to_look_at = self.navigation_camera_segmentation
        elif which_camera == "manip":
            frame = self.manipulation_camera
            masks_to_look_at = self.manipulation_camera_segmentation
        else:
            raise ValueError(f"Invalid camera: {which_camera}")

        assert (
            point[0] < frame.shape[0]
            and point[1] < frame.shape[1]
            and point[0] >= 0
            and point[1] >= 0
        ), f"Point {point} is not in frame shape {frame.shape}"
        for obj_id in masks_to_look_at:
            if masks_to_look_at[obj_id][point[0], point[1]]:
                return obj_id
        raise ValueError(f"No object found at pixel {point}")

    def get_objects_in_hand_sphere(self):
        return self.controller.last_event.metadata["arm"]["pickupableObjects"]

    def get_held_objects(self):
        return self.controller.last_event.metadata["arm"]["heldObjects"]

    def get_arm_sphere_center(self):
        return self.controller.last_event.metadata["arm"]["handSphereCenter"]

    def get_wrist_center(self):
        wrist_center = self.controller.last_event.metadata["arm"]["joints"][-2]
        assert wrist_center["name"] == "stretch_robot_wrist_1_jnt"
        return wrist_center["position"]

    def dist_from_arm_to_obj(self, object_id):
        object_location = [
            self.get_object_position(object_id)[k] for k in ["x", "y", "z"]
        ]
        arm_location = self.get_arm_wrist_absolute_position()
        return (
            (torch.Tensor(arm_location) - torch.Tensor(object_location)).norm().item()
        )

    def dist_from_arm_sphere_center_to_obj(self, object_id):
        return position_dist(
            self.get_object_position(object_id),
            self.get_arm_sphere_center(),
            ignore_y=False,
        )

    def dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(self, object_id):
        arm_sphere_center = self.get_arm_sphere_center()
        points_on_obj = self.controller.step(
            action="PointOnObjectsCollidersClosestToPoint",
            objectId=object_id,
            point=arm_sphere_center,
        ).metadata["actionReturn"]
        if points_on_obj is None or len(points_on_obj) == 0:
            return self.dist_from_arm_sphere_center_to_obj(object_id)
        else:
            dists = [
                position_dist(arm_sphere_center, p, ignore_y=False)
                for p in points_on_obj
            ]
        return min(dists)

    def get_floor_level(self):
        return (
            self.controller.last_event.metadata["agent"]["position"]["y"]
            - AGENTS_BASE_HEIGHT
        )

    @property
    def navigation_camera(self):
        frame = self.controller.last_event.frame
        cutoff = round(
            frame.shape[1] * STRETCH_CAMERA_HORIZONTAL_CROP / self.camera_width
        )
        return frame[:, cutoff:-cutoff, :]

    def get_cutoff_amount(self):
        frame = self.controller.last_event.frame
        cutoff = round(
            frame.shape[1] * STRETCH_CAMERA_HORIZONTAL_CROP / self.camera_width
        )
        return cutoff

    @property
    def manipulation_camera(self):
        frame = self.controller.last_event.third_party_camera_frames[0]
        cutoff = round(
            frame.shape[1] * STRETCH_CAMERA_HORIZONTAL_CROP / self.camera_width
        )
        return frame[:, cutoff:-cutoff, :3]  # Drop any alpha channel after cropping.

    @property
    def navigation_camera_segmentation(
        self,
    ):  # This returns the uncropped camera mask.
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a navigation_camera_segmentation"
            )

        return self.controller.last_event.instance_masks

    @property
    def manipulation_camera_segmentation(
        self,
    ):  # This returns the uncropped camera mask.
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a manipulation_camera_segmentation"
            )

        return self.controller.last_event.third_party_instance_masks[0]

    def get_segmentation_mask_of_object(
        self, object_id: str, which_camera: Literal["nav", "manip"]
    ):
        if which_camera == "nav":
            segmentation_to_look_at = self.navigation_camera_segmentation
        elif which_camera == "manip":
            segmentation_to_look_at = self.manipulation_camera_segmentation
        else:
            raise NotImplementedError

        if object_id in segmentation_to_look_at:
            mask = segmentation_to_look_at[object_id]
            cutoff = round(
                mask.shape[1] * STRETCH_CAMERA_HORIZONTAL_CROP / self.camera_width
            )
            result = mask[:, cutoff:-cutoff]
            assert result.shape == self.navigation_camera.shape[:2]
            return result
        else:
            return np.zeros(self.navigation_camera.shape[:2], dtype=bool)

    def get_relative_stretch_current_arm_state(self):
        arm = self.controller.last_event.metadata["arm"]["joints"]
        z = arm[-1]["rootRelativePosition"]["z"]
        x = arm[-1]["rootRelativePosition"]["x"]
        assert abs(x - 0) < 1e-3
        y = arm[0]["rootRelativePosition"]["y"] - 0.16297650337219238
        return dict(x=x, y=y, z=z)

    def step(self, **kwargs):
        if (
            kwargs.get("action") == "RotateAgent"
            and self.rotation_noise_std_degrees > 0
        ):
            kwargs["degrees"] += np.random.normal(0.0, self.rotation_noise_std_degrees)

        if "renderImageSynthesis" not in kwargs:
            kwargs["renderImageSynthesis"] = self.should_render_image_synthesis

        if kwargs["action"] in ["Teleport", "TeleportFull"]:
            # We don't want users to call teleport directly because this can mess up the camera horizon
            raise NotImplementedError(
                f"Use `teleport_agent` instead of `step` for teleportation (attempted action: {kwargs['action']})."
            )

        if kwargs["action"] == "__Teleport__":
            # This is how we allow the stretch agent itself to call Teleport itself without raising an error
            kwargs["action"] = "Teleport"

        return self.controller.step(**kwargs)

    def teleport_agent(
        self, position: Vector3, rotation: Union[Vector3, float], **kwargs
    ) -> Event:
        if isinstance(rotation, Dict):
            rotation = rotation["y"]

        if "standing" in kwargs:
            del kwargs["standing"]

        if "horizon" in kwargs:
            del kwargs["horizon"]
            # warnings.warn(
            #     "`horizon` is not a valid argument for teleport_agent, as camera locations are set on reset."
            #     " This argument will be ignored."
            # )

        if len(kwargs) > 0:
            allowed_keys = {
                "forceAction",
                "renderImage",
                "renderImageSynthesis",
                "raise_for_failure",
                "agentId",
            }
            assert set(kwargs.keys()).issubset(allowed_keys), (
                f"Invalid arguments for teleport_agent: {set(kwargs.keys()) - allowed_keys}"
            )

        return self.step(
            action="__Teleport__",
            position=position,
            rotation=dict(x=0, y=rotation, z=0),
            **kwargs,
        )

    def reset_visibility_cache(self):
        self._nav_visible_objects_cache = {}
        self._manip_visible_objects_cache = {}

    def get_top_down_path_view(self, agent_path, targets_to_highlight=None):
        if len(self.controller.last_event.third_party_camera_frames) < 2:
            event = self.controller.step({"action": "GetMapViewCameraProperties"})
            cam = event.metadata["actionReturn"].copy()
            bounds = event.metadata["sceneBounds"]["size"]
            max_bound = max(bounds["x"], bounds["z"])

            cam["fieldOfView"] = 50
            cam["position"]["y"] += 1.1 * max_bound
            cam["orthographic"] = False
            cam["farClippingPlane"] = 50
            del cam["orthographicSize"]
            self.controller.step(
                {"action": "AddThirdPartyCamera", "skyboxColor": "white", **cam}
            )

        waypoints = []
        for target in targets_to_highlight or []:
            target_position = self.get_object_position(target)
            target_dict = {
                "position": target_position,
                "color": {"r": 1, "g": 0, "b": 0, "a": 1},
                "radius": 0.5,
                "text": "",
            }
            waypoints.append(target_dict)

        event = self.controller.step(
            {
                "action": "VisualizeWaypoints",
                "waypoints": waypoints,
            }
        )
        # put this over the waypoints just in case
        event = self.controller.step(
            {"action": "VisualizePath", "positions": agent_path, "pathWidth": 0.2}
        )
        self.controller.step({"action": "HideVisualizedPath"})
        path = event.third_party_camera_frames[-1]
        cutoff = round(
            path.shape[1] * STRETCH_CAMERA_HORIZONTAL_CROP / self.camera_width
        )
        return path[:, cutoff:-cutoff, :]

    def noise_in_camera_horizon(self):
        return random.choice(np.arange(-2, 2, 0.2))

    def noise_in_camera_fov(self):
        return random.choice(np.arange(-1, 1, 0.1))

    def calibrate_agent(self):
        # self.teleport_agent(horizon=0, standing=True)
        self._current_horizon = 27.0 + self.noise_in_camera_horizon()
        self.step(
            action="RotateCameraMount",
            degrees=self._current_horizon,
            secondary=False,
            raise_for_failure=True,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + self.noise_in_camera_fov(),
            camera="FirstPersonCharacter",
            raise_for_failure=True,
        )
        self.step(
            action="RotateCameraMount",
            degrees=33.0 + self.noise_in_camera_horizon(),
            secondary=True,
            raise_for_failure=True,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + self.noise_in_camera_fov(),
            camera="SecondaryCamera",
            raise_for_failure=True,
        )
        self.step(
            action="SetGripperOpenness",
            openness=30,
            raise_for_failure=True,
        )

    def reset(self, scene):
        if scene is None:
            raise ValueError("`scene` must be non-None.")

        self.current_scene_json = scene
        self.agent_ids = [i for (i, r) in AGENT_RADIUS_LIST]

        # add metadata here for navmesh?
        base_agent_navmesh = {
            "agentHeight": 1.8,
            "agentSlope": 10,
            "agentClimb": 0.5,
            "voxelSize": 0.1666667,
        }
        scene["metadata"]["navMeshes"] = [
            {**base_agent_navmesh, **{"id": i, "agentRadius": r}}
            for (i, r) in AGENT_RADIUS_LIST
        ]

        scene["metadata"]["agent"]["horizon"] = HORIZON

        self.reset_visibility_cache()

        reset_event = self.controller.reset(scene=scene)

        self.calibrate_agent()

        # Do not display the unrealistic blue sphere on the agent's gripper
        self.controller.step(
            "ToggleMagnetVisibility", visible=False, raise_for_failure=True
        )

        self.set_object_filter([])

        self.room_poly_map, self.room_type_dict = get_rooms_polymap_and_type(
            self.current_scene_json
        )

        teleport_event = self.teleport_agent(**scene["metadata"]["agent"])

        if not teleport_event.metadata["lastActionSuccess"]:
            print("FAILED TO TELEPORT AGENT AFTER INITIALIZATION", scene)
            return teleport_event

        return reset_event

    def get_all_camera_parameters(self) -> Dict:
        """
        Returns a dictionary with the camera parameters for the navigation and manipulation cameras
        Each camera has position, rotation and field of view
        """
        navigation_camera_param = dict(
            position=self.controller.last_event.metadata[
                "agentPositionRelativeCameraPosition"
            ],
            rotation=self.controller.last_event.metadata[
                "agentPositionRelativeCameraRotation"
            ],
            fov=self.controller.last_event.metadata["fov"],
        )
        third_party_values = self.controller.last_event.metadata["thirdPartyCameras"][0]
        manipulation_camera_param = dict(
            position=third_party_values[
                "agentPositionRelativeThirdPartyCameraPosition"
            ],
            rotation=third_party_values[
                "agentPositionRelativeThirdPartyCameraRotation"
            ],
            fov=third_party_values["fieldOfView"],
        )
        return dict(
            navigation_camera=navigation_camera_param,
            manipulation_camera=manipulation_camera_param,
        )

    # removed to induce errors for moving to new get_objects api
    # def get_all_objects_of_type(self, object_type):
    #     with self.include_object_metadata_context():
    #         return self.controller.last_event.objects_by_type(object_type)

    def get_visible_objects(
        self,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
    ):
        # FYI: filtering by objects at this level has been removed to make best use
        # of the cache, but GetVisibleObjects still supports it with a list passed as objectIds=filter_object_ids.

        assert which_camera in ["nav", "manip", "both"]

        # Use the appropriate cache if it's available
        if (
            which_camera == "nav"
            and maximum_distance in self._nav_visible_objects_cache
        ):
            return self._nav_visible_objects_cache[maximum_distance]
        elif (
            which_camera == "manip"
            and maximum_distance in self._manip_visible_objects_cache
        ):
            return self._manip_visible_objects_cache[maximum_distance]
        elif (
            maximum_distance in self._nav_visible_objects_cache
            and maximum_distance in self._manip_visible_objects_cache
        ):
            return (
                self._nav_visible_objects_cache[maximum_distance]
                + self._manip_visible_objects_cache[maximum_distance]
            )

        visible_objects = set()
        if which_camera in ["nav", "both"]:
            if maximum_distance not in self._nav_visible_objects_cache:
                nav_visible_objects = self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    # objectIds=filter_object_ids,
                ).metadata["actionReturn"]
                self._nav_visible_objects_cache[maximum_distance] = nav_visible_objects
            else:
                nav_visible_objects = self._nav_visible_objects_cache[maximum_distance]
            visible_objects.update(nav_visible_objects)

        if which_camera in ["manip", "both"]:
            if maximum_distance not in self._manip_visible_objects_cache:
                manip_visible_objects = self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    thirdPartyCameraIndex=0,
                ).metadata["actionReturn"]
                self._manip_visible_objects_cache[maximum_distance] = (
                    manip_visible_objects
                )
            else:
                manip_visible_objects = self._manip_visible_objects_cache[
                    maximum_distance
                ]
            visible_objects.update(manip_visible_objects)

        return list(visible_objects)

    def get_approx_object_mask(
        self, object_id: str, which_camera: Literal["nav", "manip"], divisions: int
    ):
        step_dict = dict(
            action="GetApproxObjectMask",
            objectId=object_id,
            # thirdPartyCameraIndex=None if which_camera == "nav" else 0,
            divisions=divisions,
        )
        if which_camera == "manip":
            step_dict["thirdPartyCameraIndex"] = 0
        return self.step(**step_dict).metadata["actionReturn"]

    def object_is_visible_in_camera(
        self,
        object_id,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
    ):
        return object_id in self.get_visible_objects(
            which_camera=which_camera,
            maximum_distance=maximum_distance,
        )

    def get_objects(self) -> List[SimObject]:
        with self.include_object_metadata_context():
            return [
                SimObject(o) for o in self.controller.last_event.metadata["objects"]
            ]

    def get_synset_and_pos_dict(self, uninteresting_synsets: Set[str]):
        all_obj = {}
        for obj in self.get_objects():
            if obj["synset"] in uninteresting_synsets:
                continue
            all_obj[obj["objectId"]] = {
                "type": obj["synset"],
                "pos": obj["position"],
            }
        return all_obj

    def set_object_filter(self, object_ids: List[str]):
        assert len(object_ids) == 0, (
            "Only an empty object filter is supported by the release pipeline."
        )
        return self.controller.step(
            action="SetObjectFilter",
            objectIds=object_ids,
            raise_for_failure=True,
        )

    def reset_object_filter(self):
        return self.controller.step(action="ResetObjectFilter")

    @contextmanager
    def include_object_metadata_context(self):
        needs_reset = len(self.controller.last_event.metadata["objects"]) == 0
        try:
            if needs_reset:
                self.controller.step("ResetObjectFilter")
                assert self.controller.last_event.metadata["lastActionSuccess"]
            yield None
        finally:
            if needs_reset:
                obj_meta = self.controller.last_event.metadata["objects"]
                self.controller.step("SetObjectFilter", objectIds=[])
                self.controller.last_event.metadata["objects"] = obj_meta
                assert self.controller.last_event.metadata["lastActionSuccess"]

    def get_objects_that_objects_are_on(
        self, object_ids: Sequence[str]
    ) -> Dict[str, Optional[str]]:
        oid_to_on_oids = self.controller.step(
            action="CheckWhatObjectsOn",
            belowDistance=0.05,
            objectIds=object_ids,
            raise_for_failure=True,
        ).metadata["actionReturn"]

        on_oids = list(
            set(
                sum(
                    [
                        on_oid
                        for on_oid in oid_to_on_oids.values()
                        if on_oid is not None
                    ],
                    [],
                )
            )
        )

        on_oid_to_object = {None: None}
        if len(on_oids) != 0:
            on_oid_metadata = self.controller.step(
                action="GetMinimalObjectMetadata",
                objectIds=on_oids,
                raise_for_failure=True,
            ).metadata["actionReturn"]
            on_oid_to_object.update(
                {md["objectId"]: SimObject(md) for md in on_oid_metadata}
            )

        return {
            oid: [on_oid_to_object[on_oid] for on_oid in on_oids]
            for oid, on_oids in oid_to_on_oids.items()
        }

    def get_object_receptacle_synsets(self, object_id: str):
        """Return parent-receptacle synsets for an object.

        This queries receptacle metadata and should not be called every step.
        """
        source_receptacle_ids = self.get_object(
            object_id, include_receptacle_info=True
        )["parentReceptacles"]

        if source_receptacle_ids is None:
            source_receptacle_ids = []

        source_receptacle_synsets = [
            self.get_object(obj_id, include_receptacle_info=True)["synset"]
            for obj_id in source_receptacle_ids
        ]
        return source_receptacle_synsets

    def get_locations_on_receptacle(self, receptacle_id):
        result = self.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=True,
        )
        return result.metadata["actionReturn"]

    def get_current_agent_position(self):
        return StretchState(self.controller).base_position

    def get_current_agent_full_pose(self):
        return {
            **self.controller.last_event.metadata["agent"],
            "arm": self.controller.last_event.metadata["arm"],
        }

    def query_env(self, **kwargs):
        """
        :param kwargs: action, and other arguments to query the controller for information
        :return: Metadata from the environment
        """

        if "action" in kwargs:
            output = self.controller.step(**kwargs).metadata["actionReturn"]
        else:
            raise NotImplementedError
        return output

    def get_objects_of_synset_list(
        self,
        target_object_synsets: Iterable[str],
        include_hyponyms: bool,
        all_objs: Optional[List[SimObject]] = None,
    ):
        if all_objs is None:
            all_objs = self.get_objects()

        if include_hyponyms:
            return [
                sim_object
                for sim_object in all_objs
                if any(
                    is_hypernym_of(synset=sim_object["synset"], possible_hypernym=other)
                    for other in target_object_synsets
                )
            ]
        else:
            return [
                sim_object
                for sim_object in all_objs
                if sim_object["synset"] in target_object_synsets
            ]

    def get_all_objects_of_synset(
        self,
        synset: str,
        include_hyponyms: bool,
        all_objs: Optional[List[SimObject]] = None,
    ):
        return self.get_objects_of_synset_list(
            target_object_synsets=[synset],
            include_hyponyms=include_hyponyms,
            all_objs=all_objs,
        )

    def get_available_object_synsets_from_synset_list(
        self,
        target_object_synsets: Iterable[str],
        include_hyponyms: bool,
        all_objs: Optional[List[SimObject]] = None,
    ) -> Set[str]:
        return {
            o["synset"]
            for o in self.get_objects_of_synset_list(
                target_object_synsets=target_object_synsets,
                include_hyponyms=include_hyponyms,
                all_objs=all_objs,
            )
        }

    def get_object(self, object_id: str, include_receptacle_info: bool = False):
        """
        NOTE: It may be much less efficient to `include_receptacle_info` than to not.

        :param object_id:
        :param include_receptacle_info:
        :return:
        """
        if include_receptacle_info or any(
            object_id == o["objectId"]
            for o in self.controller.last_event.metadata["objects"]
        ):
            with self.include_object_metadata_context():
                return SimObject(self.controller.last_event.get_object(object_id))

        meta = self.controller.step(
            action="GetObjectMetadata", objectIds=[object_id], raise_for_failure=True
        ).metadata["actionReturn"][0]

        del meta[
            "parentReceptacles"
        ]  # This will always be None when using GetObjectMetadata so remove it so there is no ambiguity
        return SimObject(meta)

    def get_obj_pos_from_obj_id(self, object_id):
        return self.get_object(object_id)["axisAlignedBoundingBox"]["center"]

    def get_object_position(self, object_id):
        try:
            return self.get_object(object_id)["position"]
        except Exception:
            event = self.get_object(object_id)
            print(event)
            print(object_id)

    def get_agent_alignment_to_object(
        self, object_id: str, use_arm_orientation: bool = False
    ):
        current_agent_pose = StretchState(self)
        alignment = agent_alignment_to_point(
            current_agent_pose,
            self.get_object_position(object_id),
            arm=use_arm_orientation,
        )
        return alignment

    def get_agent_alignment_to_wall(self, wall_id, use_arm_orientation: bool = False):
        current_agent_pose = StretchState(self)
        wall_location = get_wall_center_floor_level(
            wall_id, y=current_agent_pose.base_position["y"]
        )
        return agent_alignment_to_point(
            current_agent_pose, wall_location, arm=use_arm_orientation
        )

    def get_reachable_positions(self, grid_size: Optional[float] = None):
        if grid_size is None:
            # Use a smaller grid size than the default as otherwise we may miss many
            # positions that are reachable when not moving with 90 degree rotations
            grid_size = GRID_SIZE * 0.75

        rp_event = self.controller.step(
            action="GetReachablePositions", gridSize=grid_size
        )
        if not rp_event:
            # NOTE: Skip scenes where GetReachablePositions fails
            warnings.warn(f"GetReachablePositions failed in {self.current_scene_json}")
            return []
        reachable_positions = rp_event.metadata["actionReturn"]
        return reachable_positions

    def get_touching_poses(
        self,
        object_id: str,
        positions: Optional[List[Vector3]] = None,
        max_distance: float = 1.0,
        max_poses: Optional[int] = None,
    ):
        other_action_kwargs = {}
        if max_poses is not None:
            other_action_kwargs["maxPoses"] = max_poses

        tp_event = self.controller.step(
            action="GetTouchingPoses",
            objectId=object_id,
            positions=positions or self.get_reachable_positions(),
            maxDistance=max_distance,
            **other_action_kwargs,
        )
        if not tp_event:
            warnings.warn("GetTouchingPoses failed")
            return []

        return tp_event.metadata["actionReturn"]

    def stop(self):
        self.controller.stop()

    def sufficient_agent_state_change(
        self, agent_state_before: StretchState, agent_state_after: StretchState
    ):
        # get the absolute value differences between the keys of the two states
        too_small, _ = StretchState.state_change_within_tolerance(
            delta_state=StretchState.difference(
                final_state=agent_state_after, initial_state=agent_state_before
            ),
            tolerance=self._universal_state_tolerance,
        )
        return not too_small

    def agent_step(self, action: StretchAction):
        agents_full_pose_before_action = StretchState(self.controller)

        action_dicts = action.enact()
        # this is a list of dicts that all resolve a dimension of motion in the environment
        # (e.g. base translate, arm up, wrist rotate, etc.)
        # For this pipeline, this is always a list of a single action
        # I have strong feelings about preserving the flexibility to actuate
        # multiple dimensions "at once" (given THOR limitations)
        # but I'm not absolutely fixated on this exact structure
        # let's talk
        all_sim_API_action_strings = [
            action_dict["action"] for action_dict in action_dicts
        ]
        for action_dict in action_dicts:
            if action_dict["action"] in [
                "RotateWristRelative",
                "MoveArm",
                "MoveArmRelative",
            ]:
                action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
            elif action_dict["action"] == "MoveAgent":
                action_dict = {**action_dict, **ADDITIONAL_NAVIGATION_ARGS}

            event = self.step(**action_dict)

        if "ReleaseObject" in all_sim_API_action_strings:
            self.step(action="AdvancePhysicsStep", simSeconds=2)

        agents_full_pose_after_action = StretchState(self.controller)

        agent_moved = self.sufficient_agent_state_change(
            agents_full_pose_before_action, agents_full_pose_after_action
        )
        collision_in_error_message = (
            "collided" in event.metadata["errorMessage"].lower()
        )
        if isinstance(action, StretchGraspAction):
            if len(agents_full_pose_after_action.held_oids) > len(
                agents_full_pose_before_action.held_oids
            ):
                action_success = True
            else:
                action_success = False
        elif isinstance(action, StretchDropOffAction):
            action_success = True
        elif any(
            "arm" in action.lower() or "wrist" in action.lower()
            for action in all_sim_API_action_strings
        ):
            action_success = not collision_in_error_message and agent_moved
        else:
            action_success = not collision_in_error_message

        event.metadata["lastActionSuccess"] = action_success

        return event

    # calculate the shortest path to that location
    def get_shortest_path_to_object(
        self,
        object_id,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement: bool = True,
    ) -> Optional[List[Vector3]]:
        """
        Computes the shortest path to an object from an initial position using a controller

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        mesh_restriction = specific_agent_meshes is not None
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids

        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPath",
                objectId=object_id,
                position=initial_position,
                navMeshIds=[nav_mesh_id],  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue

                if (
                    nav_mesh_id > 1
                    and not mesh_restriction
                    and attempt_path_improvement
                    and len(corners) > 4
                ):
                    corners = self.split_and_replan_paths(
                        initial_position, corners[-1], corners, recursion_depth=1
                    )
                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    def does_some_shortest_path_to_object_exist(
        self,
        object_id: str,
        initial_position=None,
        initial_rotation=None,
    ) -> bool:
        """
        Checks if a shortest path to an object from an initial position exists. This is faster than
        `get_shortest_path_to_object` as we will only use the most general nav mesh.

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        return (
            self.get_shortest_path_to_object(
                object_id=object_id,
                initial_position=initial_position,
                initial_rotation=initial_rotation,
                specific_agent_meshes=[self.agent_ids[-1]],
                attempt_path_improvement=False,
            )
            is not None
        )

    def split_and_replan_paths(
        self, initial_position, target_position, path, recursion_depth=0
    ):
        first_half = path[: (len(path) // 2)]
        second_half = path[(len(path) // 2) :]

        # Recursive call to get_shortest_path_to_point for each half
        first_half_replan = self.get_shortest_path_to_point(
            first_half[-1], initial_position, recursion_depth=recursion_depth
        )
        if first_half_replan is not None:
            first_half = first_half_replan

        second_half_replan = self.get_shortest_path_to_point(
            target_position, first_half_replan[-1], recursion_depth=recursion_depth
        )
        if second_half_replan is not None:
            second_half = second_half_replan
        return first_half + second_half[1:]

    # calculate the shortest path to that location
    def get_shortest_path_to_point(
        self,
        target_position,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement=True,
        recursion_depth=0,
    ):
        """
        Computes the shortest path to an object from an initial position using a controller
        :param controller: agent controller
        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        mesh_restriction = specific_agent_meshes is not None
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPathToPoint",
                position=initial_position,
                target=target_position,
                navMeshIds=[nav_mesh_id],  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                if (
                    nav_mesh_id > 1
                    and not mesh_restriction
                    and attempt_path_improvement
                    and len(corners) > 4
                    and recursion_depth < 3
                ):
                    corners = self.split_and_replan_paths(
                        initial_position, target_position, corners, recursion_depth + 1
                    )

                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    def num_pixels_visible(self, object_id, manipulation_camera=False):
        assert (
            "renderInstanceSegmentation" in self.initialization_args
            and self.initialization_args["renderInstanceSegmentation"]
        )
        if manipulation_camera:
            masks = self.manipulation_camera_segmentation
        else:
            masks = self.navigation_camera_segmentation

        if object_id not in masks:
            return 0

        mask = masks[object_id]
        return mask.sum()

    def is_object_visible_enough_for_interaction(
        self, object_id: str, manipulation_camera=True
    ):
        return is_any_object_sufficiently_visible_and_in_center_frame(
            controller=self,
            object_ids=[object_id],
            manipulation_camera=manipulation_camera,
            object_synset=None,
        )

    def get_closest_object_from_ids(self, object_ids, return_id_and_dist: bool = False):
        all_paths = [
            (
                obj_id,
                self.get_shortest_path_to_object(
                    obj_id,
                    specific_agent_meshes=[self.agent_ids[-1]],
                    attempt_path_improvement=False,
                ),
            )
            for obj_id in object_ids
        ]

        min_dist = float("inf")
        closest_obj_id = None
        for obj_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id
        return closest_obj_id if not return_id_and_dist else (closest_obj_id, min_dist)

    def get_candidate_points_in_room(
        self,
        room_id,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        polygon = self.room_poly_map[room_id]

        if room_triangles is None:
            # Triangulates the room, and takes the centers of all triangles as possible
            # target locations
            room_triangles = triangulate_room_polygon(polygon)

        candidate_points = [
            ((t.centroid.x, t.centroid.y), t.area)
            for t in room_triangles  # type:ignore
        ]

        # We sort the triangles by size so we try to go to the center of the largest triangle first
        candidate_points.sort(key=lambda x: x[1], reverse=True)
        candidate_points = [p[0] for p in candidate_points]

        # The centroid of the whole room polygon need not be in the room when the room is concave. If it is,
        # let's make it the first point we try to navigate to.
        if polygon.contains(polygon.centroid):
            candidate_points.insert(0, (polygon.centroid.x, polygon.centroid.y))

        candidate_points = [
            p
            for p in candidate_points
            if self.room_poly_map[room_id].contains(Point(p))
        ]

        return candidate_points

    def get_shortest_path_to_room_candidate_points(
        self,
        candidate_points,
        specific_agent_meshes=None,
        max_tries: int = 5,
        use_largest_possible_mesh: bool = False,
    ):
        assert max_tries > 0

        current_agent_position = self.controller.last_event.metadata["agent"][
            "position"
        ]
        y = current_agent_position["y"]

        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        specific_agent_meshes = sorted(specific_agent_meshes)

        path = None
        for agent_id in specific_agent_meshes:
            for point in candidate_points[:max_tries]:
                path = self.get_shortest_path_to_point(
                    target_position=dict(x=point[0], y=y, z=point[1]),
                    initial_position=current_agent_position,
                    specific_agent_meshes=[agent_id],
                    attempt_path_improvement=False,
                )
                if path is not None:
                    break
            if use_largest_possible_mesh and path is not None:
                break
        return path

    def get_shortest_path_to_room(
        self,
        room_id,
        specific_agent_meshes=None,
        max_tries: int = 5,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        candidate_points = self.get_candidate_points_in_room(
            room_id=room_id,
            room_triangles=room_triangles,
        )

        return self.get_shortest_path_to_room_candidate_points(
            candidate_points, specific_agent_meshes, max_tries
        )

    def get_objects_room_id_and_type(self, object_id: str) -> Tuple[str, str]:
        object_position = self.get_object_position(object_id)
        room_id = get_room_id_from_location(self.room_poly_map, object_position)
        room_type_return = (
            self.room_type_dict[room_id] if room_id is not None else None
        )  # making it more robust to none style cases
        return room_id, room_type_return

    def find_closest_room_of_list(self, room_ids, return_id_and_dist: bool = False):
        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        min_dist = float("inf")
        closest_room_id = None
        for room_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_room_id = room_id

        return (
            closest_room_id if not return_id_and_dist else (closest_room_id, min_dist)
        )

    def get_current_scene_json(self):
        return self.current_scene_json

    def get_agent_dist_from_room_ids(
        self,
        rooms,
    ):
        room_ids = [room["id"] for room in rooms]

        [room["id"] for room in self.current_scene_json["rooms"]]

        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        to_return = {}
        for room_id, path in all_paths:
            if path is None:
                continue
            to_return[room_id] = sum_dist_path(path)

        return to_return
