import argparse
from .utils.arg_parsers import parsers  # pylint: disable=g-bad-import-order

class ResnetArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model.
  """

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.PerformanceParser(),
        parsers.ImageModelParser(),
    ])

    self.add_argument(
        '--version', '-v',
        type=int,
        choices=[1, 2],
        default=1,
        help="Version of ResNet. (1 or 2) See README.md for details."
    )

    self.add_argument(
        '--resnet_size', '-rs', type=int, default=50,
        choices=resnet_size_choices,
        help='[default: %(default)s] The size of the ResNet model to use.',
        metavar='<RS>' if resnet_size_choices is None else None
    )
