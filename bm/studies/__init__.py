# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# all studies should be imported so as to populate the recordings dictionary
from . import brennan2019  # noqa
from . import broderick2019  # noqa
from . import fake  # noqa
from . import gwilliams2022  # noqa
from . import lebel2023  # noqa
from . import schoffelen2019  # noqa

# flake8: noqa
from .api import Recording, from_selection, register
