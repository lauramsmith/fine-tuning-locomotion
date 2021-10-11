"""Class to read in motion capture data and output position and orientation."""

from typing import Sequence, Text

import numpy as np
from phasespace import grpc_stream_client
from pybullet_utils import transformations

_ARRAY = Sequence[float]

MILLIMETER_TO_METER = 0.001
# ID that Phasespace has assigned to each marker.
FRONT_LEFT_ID = 3
FRONT_RIGHT_ID = 7
BACK_LEFT_ID = 5
BACK_RIGHT_ID = 1
MARKER_DICT = {
    FRONT_LEFT_ID: 0,
    FRONT_RIGHT_ID: 1,
    BACK_LEFT_ID: 2,
    BACK_RIGHT_ID: 3
}


class PhaseSpaceRobotTracker(object):
  """Reads in motion capture data and outputs position and orientation."""

  def __init__(self, server: Text = "localhost:12345"):
    """Constructor.

    Args:
      server: Hostname and port of the gRPC server outputting marker data protos
        (e.g. "localhost:12345").
    """
    self._client = grpc_stream_client.GrpcStreamClient(server)

    # If the markers have been configured as a rigid body, set this.
    self.rigid = None

    num_markers = 4  # Assume four markers are attached to the robot's base.
    # (x, y, z) -> 3 coordinates for each marker.
    self._current_marker_positions = np.zeros(num_markers * 3)

  def update(self) -> None:
    """Push the current marker positions to the last and update the current."""
    marker_data = self._client.get_marker_data_adhoc()
    if marker_data is not None:
      if marker_data.rigids:
        self.rigid = marker_data.rigids[0]
      else:
        self._update_current_marker_positions(marker_data)

  def _update_current_marker_positions(self, marker_data):
    """Updates the current marker positions by marker indices."""

    for marker in marker_data.markers:
      index = MARKER_DICT[marker.marker_id] * 3
      pos = np.array((marker.position_mm.x, marker.position_mm.y,
                      marker.position_mm.z)) * MILLIMETER_TO_METER
      self._current_marker_positions[index:index + len(pos)] = pos

  def get_base_position(self) -> _ARRAY:
    """Returns the base position of the robot in meters."""
    if self.rigid:
      pos = np.array((self.rigid.position_mm.x, self.rigid.position_mm.y,
                      self.rigid.position_mm.z)) * MILLIMETER_TO_METER
    else:
      pos = np.mean(self._current_marker_positions.reshape(-1, 3), axis=0)
    return pos

  def _get_rotation_matrix(self) -> _ARRAY:
    """Returns robot's current orientation."""
    front_left_id = MARKER_DICT[FRONT_LEFT_ID]
    front_right_id = MARKER_DICT[FRONT_RIGHT_ID]
    back_left_id = MARKER_DICT[BACK_LEFT_ID]
    back_right_id = MARKER_DICT[BACK_RIGHT_ID]
    assert self._current_marker_positions.shape[0] >= 4 * 3

    front_left_pos = self._current_marker_positions[
        (front_left_id * 3):((front_left_id + 1) * 3)]
    front_right_pos = self._current_marker_positions[
        (front_right_id * 3):((front_right_id + 1) * 3)]
    back_left_pos = self._current_marker_positions[
        (back_left_id * 3):((back_left_id + 1) * 3)]
    back_right_pos = self._current_marker_positions[
        (back_right_id * 3):((back_right_id + 1) * 3)]

    forward = 0.5 * (front_left_pos + front_right_pos) \
        - 0.5 * (back_left_pos + back_right_pos)

    left = 0.5 * (front_left_pos + back_left_pos) \
        - 0.5 * (front_right_pos + back_right_pos)

    up = np.cross(forward, left)
    left = np.cross(up, forward)

    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    left /= np.linalg.norm(left)

    return np.transpose(np.array([forward, left, up]))

  def get_base_orientation(self) -> _ARRAY:
    """Returns base orientation of the robot as quaternion."""
    if self.rigid:
      return np.array((self.rigid.quat.x, self.rigid.quat.y, self.rigid.quat.z,
                       self.rigid.quat.w))
    return transformations.quaternion_from_matrix(self._get_rotation_matrix())

  def get_base_roll_pitch_yaw(self) -> _ARRAY:
    """Returns base orientation of the robot in radians."""
    if self.rigid:
      return transformations.euler_from_quaternion(self.get_base_orientation())
    return transformations.euler_from_matrix(self._get_rotation_matrix())
