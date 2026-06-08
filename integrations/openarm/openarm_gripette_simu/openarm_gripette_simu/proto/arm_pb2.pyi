from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CartesianDelta(_message.Message):
    __slots__ = ("dx", "dy", "dz", "dr6d")
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    DZ_FIELD_NUMBER: _ClassVar[int]
    DR6D_FIELD_NUMBER: _ClassVar[int]
    dx: float
    dy: float
    dz: float
    dr6d: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, dx: _Optional[float] = ..., dy: _Optional[float] = ..., dz: _Optional[float] = ..., dr6d: _Optional[_Iterable[float]] = ...) -> None: ...

class ArmCommandResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class GetArmStateRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArmState(_message.Message):
    __slots__ = ("x", "y", "z", "r6d", "joint_positions")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    R6D_FIELD_NUMBER: _ClassVar[int]
    JOINT_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    r6d: _containers.RepeatedScalarFieldContainer[float]
    joint_positions: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., r6d: _Optional[_Iterable[float]] = ..., joint_positions: _Optional[_Iterable[float]] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("joint_positions",)
    JOINT_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    joint_positions: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, joint_positions: _Optional[_Iterable[float]] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("success", "error", "cube_x", "cube_y", "cube_z")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CUBE_X_FIELD_NUMBER: _ClassVar[int]
    CUBE_Y_FIELD_NUMBER: _ClassVar[int]
    CUBE_Z_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    cube_x: float
    cube_y: float
    cube_z: float
    def __init__(self, success: bool = ..., error: _Optional[str] = ..., cube_x: _Optional[float] = ..., cube_y: _Optional[float] = ..., cube_z: _Optional[float] = ...) -> None: ...

class SuccessStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SuccessStatusResponse(_message.Message):
    __slots__ = ("goal_reached", "cube_displacement")
    GOAL_REACHED_FIELD_NUMBER: _ClassVar[int]
    CUBE_DISPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    goal_reached: bool
    cube_displacement: float
    def __init__(self, goal_reached: bool = ..., cube_displacement: _Optional[float] = ...) -> None: ...

class ArmPingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArmPingResponse(_message.Message):
    __slots__ = ("status", "uptime_seconds")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    status: str
    uptime_seconds: float
    def __init__(self, status: _Optional[str] = ..., uptime_seconds: _Optional[float] = ...) -> None: ...
