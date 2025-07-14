# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable

import numpy as np
import torch


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    it = x
    for x in it:
        if isinstance(x, Iterable) and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x


def _unflatten(flat, prototype):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(prototype, torch.Tensor):
        prototype = prototype.cpu().numpy()

    if isinstance(prototype, np.ndarray):
        idx = prototype.size
        res = np.array(flat)[:idx].reshape(prototype.shape)

        return torch.tensor(res)

    # if isinstance(prototype, collections.Iterable):
    if isinstance(prototype, Iterable):
        res = []
        for x in prototype:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat

    raise TypeError("Unsupported type in the model: {}".format(type(prototype)))


def unflatten(flat, prototype):
    """Wrapper for :func:`_unflatten`."""
    # pylint:disable=len-as-condition
    result, tail = _unflatten(flat, prototype)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return result
