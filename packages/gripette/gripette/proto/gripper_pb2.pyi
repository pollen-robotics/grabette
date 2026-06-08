from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GripperFrame(_message.Message):
    __slots__ = ("jpeg_data", "motor_state", "timestamp_ms", "sequence")
    JPEG_DATA_FIELD_NUMBER: _ClassVar[int]
    MOTOR_STATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    jpeg_data: bytes
    motor_state: MotorState
    timestamp_ms: float
    sequence: int
    def __init__(self, jpeg_data: _Optional[bytes] = ..., motor_state: _Optional[_Union[MotorState, _Mapping]] = ..., timestamp_ms: _Optional[float] = ..., sequence: _Optional[int] = ...) -> None: ...

class MotorState(_message.Message):
    __slots__ = ("motor1_position", "motor2_position")
    MOTOR1_POSITION_FIELD_NUMBER: _ClassVar[int]
    MOTOR2_POSITION_FIELD_NUMBER: _ClassVar[int]
    motor1_position: float
    motor2_position: float
    def __init__(self, motor1_position: _Optional[float] = ..., motor2_position: _Optional[float] = ...) -> None: ...

class MotorCommand(_message.Message):
    __slots__ = ("motor1_goal", "motor2_goal")
    MOTOR1_GOAL_FIELD_NUMBER: _ClassVar[int]
    MOTOR2_GOAL_FIELD_NUMBER: _ClassVar[int]
    motor1_goal: float
    motor2_goal: float
    def __init__(self, motor1_goal: _Optional[float] = ..., motor2_goal: _Optional[float] = ...) -> None: ...

class MotorCommandResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class TorqueCommand(_message.Message):
    __slots__ = ("enable",)
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    def __init__(self, enable: bool = ...) -> None: ...

class TorqueResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class ReadMotorsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ("status", "uptime_seconds")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    status: str
    uptime_seconds: float
    def __init__(self, status: _Optional[str] = ..., uptime_seconds: _Optional[float] = ...) -> None: ...
