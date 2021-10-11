"""Basic python library to receive motion capture data over GRPC.

Basic usage:
client = grpc_stream_client.GrpcStreamClient(server_address_and_port)
stream = client.get_marker_data()
for data in stream:
  # Do something with marker data in data.
  # This loop should be able to run at least at the streaming frequency (960Hz).
"""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
from absl import logging
import grpc

from phasespace import marker_data_pb2
from phasespace import marker_data_pb2_grpc

flags.DEFINE_string(
    'mocap_grpc_server', 'localhost:12345',
    'Hostname and port of the gRPC server serving marker_data protos.')
FLAGS = flags.FLAGS


class GrpcStreamClient(object):
  """Class to connect to GRPC server with MarkersData."""

  def __init__(self, server=None):
    """Creates a GrpcStreamClient instance that connects to the given server.

    Args:
      server: hostname and port (string) of the gRPC server to connect to
        (e.g. "localhost:12345").
    """
    logging.set_verbosity(logging.INFO)
    logging.info('Connecting to stream from %s', server)
    logging.warning('Using insecure GRPC channel.')
    self.channel = grpc.insecure_channel(server)

    grpc.channel_ready_future(self.channel).result()
    self.stub = marker_data_pb2_grpc.MarkerTrackerStub(self.channel)
    self.request = marker_data_pb2.MarkerTrackerParams()
    logging.info('Connected')

  def get_marker_data(self):
    """This is a generator. Every call will yield a new MarkerData object.

    This generator should be called at least as fast as the streaming frequency
    (typically 960Hz) to ensure the latest MarkerData object is returned.
    See grpc_stream_client_multiprocessing.py for an easy to use wrapper that
    automatically provides the most recent MarkerData object without hogging the
    main loop.

    Yields:
      MarkerData: A MarkerData proto streamed from the server.
    Raises:
      grpc.RpcError: When the server dies/gets disconnected.
    """
    logging.info('Getting GRPC response')
    response = self.stub.TrackMarkers(self.request)
    for markers in response:
      yield markers

  def get_marker_data_adhoc(self):
    """This function uses an alternative MarkerTracker API.

    Retrieves MarkerData on an adhoc basis, as opposed to streaming.
    Blocking until data is received from the server.

    Returns:
      A single proto of the most recent MarkerData the server has.
    """
    return self.stub.GetLatestMarkerData(self.request)


def main(argv):
  del argv
  client = GrpcStreamClient(server=FLAGS.mocap_grpc_server)
  for data in client.get_marker_data():
    print(data)

if __name__ == '__main__':
  app.run(main)
