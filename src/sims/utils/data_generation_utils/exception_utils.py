# define Python user-defined exceptions


class PlannerFailedException(Exception):
    pass


class IrrecoverablePlannerFailureDueToHouseException(PlannerFailedException):
    pass


class PlannerFailedToNavigateException(PlannerFailedException):
    "Planner Failed to Navigate"

    pass


class PlannerFailedToFindSpotOnReceptacleException(PlannerFailedException):
    "Planner Failed to Navigate"

    pass


class PlannerFailedToNavigateToRoomException(PlannerFailedException):
    "Planner Failed to Navigate To Room"

    pass


class PlannerFailedToNavigateToReceptacleException(PlannerFailedException):
    "Planner Failed to Navigate To Room"

    pass


class PlannerFailedToPickUpException(PlannerFailedException):
    "Planner Failed to Pick up"

    pass


class PlannerFailedToSeeException(PlannerFailedException):
    "Planner Failed to see the object"

    pass


class PlannerFailedToPlaceException(PlannerFailedException):
    "Planner Failed to place the object"

    pass


class TaskSamplerException(Exception):
    "Task Sampler failed to find a valid sample"

    pass


class HouseInvalidForTaskException(TaskSamplerException):
    "Task Sampler failed to find a valid sample because the house was fully impossible to generate any task for"

    pass


class ExplorationCoverageException(PlannerFailedException):
    pass


class IrrecoverableExplorationCoverageException(
    IrrecoverablePlannerFailureDueToHouseException
):
    pass


class TaskDifficultyIncorrectException(PlannerFailedException):
    pass


class TaskSamplerInInvalidStateError(TaskSamplerException):
    "Task sampler has entered some totally invalid state from which next_task calls will definitely fail."

    pass
