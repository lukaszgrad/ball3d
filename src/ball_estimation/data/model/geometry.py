"""Elementary classes to operate on geometric figures."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from functools import cached_property
from math import isclose
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    confloat,
    root_validator,
    validate_arguments,
)


Point2DSubclass = TypeVar("Point2DSubclass", bound="Point2D")


class CoordinateNotANumberError(ValueError):
    """Raised when coordinate is NaN."""


class Point2D(BaseModel, allow_inf_nan=False, frozen=True):
    """2D point in cartesian coordinate system.

    Parameters
    ----------
    x : number
        A coordinate along the X axis.

    y : number
        A coordinate along the Y axis.

    References
    ----------
    .. [1] `Class Point Documentation
       <https://docs.opencv.org/3.4/db/d4e/classcv_1_1Point__.html>`_

    """

    x: float | int
    y: float | int

    def __neg__(self) -> Point2DSubclass:
        """Reflect point in coordinate system origin."""
        return type(self)(x=-self.x, y=-self.y)

    def __round__(self, ndigits: int | None = None) -> Point2DSubclass:
        return type(self).construct(x=round(self.x, ndigits), y=round(self.y, ndigits))

    def __eq__(self, other: Point2D) -> bool:
        if not isinstance(other, Point2D):
            return NotImplemented

        return np.allclose([self.x, self.y], [other.x, other.y])

    def __add__(self, other) -> Point2DSubclass:
        other = Point2D(x=other, y=other)

        point = type(self)(x=self.x + other.x, y=self.y + other.y)

        return point

    def __radd__(self, other) -> Point2DSubclass:
        return self + other

    def __sub__(self, other) -> Point2DSubclass:
        return self + (-other)

    def __rsub__(self, other) -> Point2DSubclass:
        return -self + other

    def __mul__(self, other) -> Point2DSubclass:
        other = Point2D(x=other, y=other)

        return type(self)(x=self.x * other.x, y=self.y * other.y)

    def __rmul__(self, other) -> Point2DSubclass:
        return self * other

    def __truediv__(self, other) -> Point2DSubclass:
        other = Point2D(x=other, y=other)

        return type(self)(x=self.x / other.x, y=self.y / other.y)

    def __rtruediv__(self, other) -> Point2DSubclass:
        other = Point2D(x=other, y=other)

        return type(self)(x=other.x / self.x, y=other.y / self.y)

    def distance(self, other: Point2D) -> float:
        """Find distance to another point."""
        value = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

        return value

    def cross(self, other: Point2D):
        """Calculate cross product with another point."""
        value = np.cross([self.x, self.y], [other.x, other.y]).item()

        return value

    def dot(self, other: Point2D):
        """Calculate dot product with another point."""
        value = np.dot([self.x, self.y], [other.x, other.y])

        return value

    def hadamard(self, *others: Point2D) -> Point2DSubclass:
        """Calculate element-wise product with another points."""
        values = tuple(
            np.prod(np.vstack(tuple(other.as_tuple() for other in others)), axis=0)
        )

        return self.from_tuple(values)

    def as_tuple(self):
        return self.x, self.y

    def to_type(self, dtype: type, /) -> Point2DSubclass:
        """Cast fields to specified type."""
        return type(self).construct(x=dtype(self.x), y=dtype(self.y))

    def into_polar(self) -> tuple[NonNegativeFloat, confloat(ge=-np.pi, le=np.pi)]:
        """Convert point to polar coordinates.

        Returns
        -------
        rho : non-negative float
            A distance from the pole (radius).

        theta : float in the range [-pi, pi]
            An angle in radians (azimuth).

        """
        rho = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)

        return rho, theta

    @classmethod
    @validate_arguments
    def from_tuple(cls, x_and_y, /) -> Point2DSubclass:
        x, y = x_and_y

        return cls(x=x, y=y)

    @classmethod
    @validate_arguments
    def from_polar(cls, rho: float, theta: float) -> Point2DSubclass:
        """Instantiate point from polar coordinates.

        Parameters
        ----------
        rho : non-negative float
            A distance from the pole (radius).

        theta : float
            An angle in radians (azimuth).

        Returns
        -------
        Point2DSubclass
            An instance of the class.

        """
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)

        return cls(x=x, y=y)

    @classmethod
    @validate_arguments
    def from_parabolic(cls, sigma: float, tau: float) -> Point2DSubclass:
        """Instantiate point from parabolic coordinates.

        Parameters
        ----------
        sigma : float
            A coordinate that when fixed forms open upwards parabolas
            that share a common focus.

        tau : float
            A coordinate that when fixed forms open downwards parabolas
            that share a common focus.

        Returns
        -------
        Point2DSubclass
            An instance of the class.

        """
        x = sigma * tau
        y = 0.5 * (sigma**2 - tau**2)

        return cls(x=x, y=y)


class AspectRatioMismatchError(ValueError):
    """Raised when manually computed and expected ratios mismatch."""


Size2DSubclass = TypeVar("Size2DSubclass", bound="Size2D")


class Size2D(BaseModel, frozen=True, keep_untouched=(cached_property,)):
    """Base class for size of 2D image or rectangle.

    Parameters
    ----------
    width : non-negative number
        Side length along the X axis.

    height : non-negative number
        Side length along the Y axis.

    Attributes
    ----------
    aspect_ratio
    area
    diagonal

    References
    ----------
    .. [1] `Class Size Documentation
       <https://docs.opencv.org/3.4/d6/d50/classcv_1_1Size__.html>`_

    """

    width: float | int
    height: float | int

    def __bool__(self) -> bool:
        """Check if size is non-degenerate."""
        return not self.is_degenerate

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    def __round__(self, ndigits: int | None = None) -> Size2DSubclass:
        """Round shape with specified number of decimals."""
        width, height = round(self.width, ndigits), round(self.height, ndigits)

        return type(self).construct(width=width, height=height)

    def __eq__(self, other: Size2D) -> bool:
        if not isinstance(other, Size2D):
            return NotImplemented

        return np.allclose([self.width, self.height], [other.width, other.height])

    def __add__(self, other) -> Size2DSubclass:
        """Expand size with another size."""
        other = Size2D(width=other, height=other)

        size = type(self)(
            width=self.width + other.width, height=self.height + other.height
        )

        return size

    def __radd__(self, other) -> Size2DSubclass:
        return self + other

    def __sub__(self, other) -> Size2DSubclass:
        """Shrunk size with another size."""
        other = Size2D(width=other, height=other)

        size = type(self)(
            width=self.width - other.width, height=self.height - other.height
        )

        return size

    def __mul__(self, other) -> Size2DSubclass:
        """Scale size by vector."""
        other = Point2D(x=other, y=other)

        return type(self)(width=self.width * other.x, height=self.height * other.y)

    def __rmul__(self, other) -> Size2DSubclass:
        return self * other

    def __truediv__(self, other) -> Size2DSubclass:
        """Scale size by vector with inverted elements."""
        other = Point2D(x=other, y=other)

        return type(self)(width=self.width / other.x, height=self.height / other.y)

    @cached_property
    def aspect_ratio(self) -> Fraction | None:
        """Ratio of size's width to its height."""
        return Fraction(self.width, self.height) if self.height != 0 else None

    @cached_property
    def area(self):
        """Size area."""
        return self.width * self.height

    @cached_property
    def diagonal(self):
        """Length of size diagonal."""
        return (self.width**2 + self.height**2) ** 0.5

    @cached_property
    def is_degenerate(self) -> bool:
        """Whether size is an empty area."""
        return isclose(self.area, 0, abs_tol=1e-6)

    #def even(self) -> Size2DSubclass:
    #    """Round to closest even size."""
    #    width, height = even(self.as_tuple())
    #
    #    return type(self).construct(width=width, height=height)

    #def odd(self) -> Size2DSubclass:
    #    """Round to closest odd size."""
    #    width, height = odd(self.as_tuple())
    #
    #    return type(self).construct(width=width, height=height)

    def as_tuple(self):
        return self.width, self.height

    def to_type(self, dtype: type, /) -> Size2DSubclass:
        """Cast fields to specified type."""
        return type(self).construct(width=dtype(self.width), height=dtype(self.height))

    def into_aspect(self):
        """Convert size to aspect ratio and height.

        Returns
        -------
        aspect_ratio : non-negative float
            The ratio of the size's width to its height.

        height : non-negative number
            Side length along the Y axis.

        """
        return self.aspect_ratio, self.height

    @classmethod
    @validate_arguments
    def from_tuple(
        cls, width_and_height, /
    ) -> Size2DSubclass:
        width, height = width_and_height

        return cls(width=width, height=height)


class Alignment(str, Enum):
    """Option defining object alignment on image."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    CENTER = "center"


class RectangleCornersError(ValueError):
    """Raised when corners of rectangle are incompatible."""


_BORDER_MARGIN_DEFAULT: Size2D = Size2D(width=0, height=0)

RectangleSubclass = TypeVar("RectangleSubclass", bound="Rectangle")


class Rectangle(BaseModel, frozen=True, keep_untouched=(cached_property,)):
    """Axis-aligned rectangle represented by its corners.

    Representation includes two diagonally opposite points, where one of
    them contains minimum values and the other maximum values along both
    axes.

    Parameters
    ----------
    p0 : Point2D
        Minimum values along the X and Y axes.

    p1 : Point2D
        Maximum values along the X and Y axes.

    Attributes
    ----------
    x0
    y0
    x1
    y1
    top_right
    top_left
    bottom_left
    bottom_right
    centroid
    size

    Notes
    -----
    It is assumed that the rectangle representation by corners is
    specified as a default initialization input. To instantiate an
    object from a different representation, use a corresponding
    `from_<representation-name>` class method.

    It is also assumed that the axes look like this:

    0/0---X--->
    |
    |
    Y
    |
    |
    v

    References
    ----------
    .. [1] L.Mao, `Bounding Box Encoding and Decoding in Object
       Detection
       <https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/>`_

    .. [2] `Class Rect Documentation
       <https://docs.opencv.org/java/2.4.9/org/opencv/core/Rect.html>`_

    """

    p0: Point2D
    p1: Point2D

    def __bool__(self) -> bool:
        """Check if rectangle is non-degenerate."""
        return bool(self.size)

    def __neg__(self) -> RectangleSubclass:
        """Reflect rectangle in origin.

        Reflect all points of the rectangle in the origin of the
        coordinate system.

        """
        return type(self)(p0=-self.p1, p1=-self.p0)

    def __round__(self, ndigits: int | None = None) -> RectangleSubclass:
        return type(self)(p0=round(self.p0, ndigits), p1=round(self.p1, ndigits))

    def __contains__(self, item: Point2D) -> bool:
        """Check if rectangle contains point."""
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1

        if x0 <= item.x <= x1 and y0 <= item.y <= y1:
            return True

        return False

    def __eq__(self, other: Rectangle) -> bool:
        if not isinstance(other, Rectangle):
            return NotImplemented

        return self.p0 == other.p0 and self.p1 == other.p1

    def __and__(self, other: Rectangle) -> RectangleSubclass | None:
        """Intersect two rectangles.

        If the resulting intersection is empty, return `None`.

        """
        if not isinstance(other, Rectangle):
            return NotImplemented

        x0, x1 = max(self.x0, other.x0), min(self.x1, other.x1)
        y0, y1 = max(self.y0, other.y0), min(self.y1, other.y1)

        if x0 > x1 or y0 > y1:
            return None

        return type(self)(p0=Point2D(x=x0, y=y0), p1=Point2D(x=x1, y=y1))

    def __or__(self, other: Rectangle) -> RectangleSubclass:
        """Enclose two rectangles with rectangle of minimal area."""
        if not isinstance(other, Rectangle):
            return NotImplemented

        x0, x1 = min(self.x0, other.x0), max(self.x1, other.x1)
        y0, y1 = min(self.y0, other.y0), max(self.y1, other.y1)

        return type(self)(p0=Point2D(x=x0, y=y0), p1=Point2D(x=x1, y=y1))

    def __add__(self, other: Point2D | Size2D) -> RectangleSubclass:
        if isinstance(other, Point2D):
            return self.shifted(offset=other)
        if isinstance(other, Size2D):
            return self.expanded(size=other)

        return NotImplemented

    def __radd__(self, other: Point2D | Size2D) -> RectangleSubclass:
        return self + other

    def __sub__(self, other: Point2D | Size2D) -> RectangleSubclass:
        if isinstance(other, Point2D):
            return self.shifted(offset=-other)
        if isinstance(other, Size2D):
            return self.shrunk(size=other)

        return NotImplemented

    def __rsub__(self, other: Point2D | Size2D) -> RectangleSubclass:
        return -self + other

    def __mul__(self, other: Point2D) -> RectangleSubclass:
        if not isinstance(other, Point2D):
            return NotImplemented

        return self.scaled(factor=other)

    def __rmul__(self, other: Point2D) -> RectangleSubclass:
        return self * other

    def __truediv__(self, other: Point2D) -> RectangleSubclass:
        if not isinstance(other, Point2D):
            return NotImplemented

        return self.invscaled(factor=other)

    @property
    def x0(self):
        """Minimum value along X axis."""
        return self.p0.x

    @property
    def y0(self):
        """Minimum value along Y axis."""
        return self.p0.y

    @property
    def x1(self):
        """Maximum value along X axis."""
        return self.p1.x

    @property
    def y1(self):
        """Maximum value along Y axis."""
        return self.p1.y

    @cached_property
    def top_right(self) -> Point2D:
        return Point2D.construct(x=self.x1, y=self.y0)

    @property
    def top_left(self) -> Point2D:
        return self.p0

    @cached_property
    def bottom_left(self) -> Point2D:
        return Point2D.construct(x=self.x0, y=self.y1)

    @property
    def bottom_right(self) -> Point2D:
        return self.p1

    @cached_property
    def centroid(self) -> Point2D:
        centroid = Point2D(
            x=self.x0 + self.size.width / 2, y=self.y0 + self.size.height / 2
        )

        return centroid

    @cached_property
    def size(self) -> Size2D:
        return Size2D(width=self.x1 - self.x0, height=self.y1 - self.y0)

    def shifted(self, offset: Point2D) -> RectangleSubclass:
        """Shift rectangle by certain offset."""
        return type(self)(p0=self.p0 + offset, p1=self.p1 + offset)

    def expanded(self, size: Size2D) -> RectangleSubclass:
        """Expand rectangle by certain size."""
        return self.from_centroid(centroid=self.centroid, size=self.size + size)

    def shrunk(self, size: Size2D) -> RectangleSubclass:
        """Shrink rectangle by certain size.

        If a dimension of `size` exceeds the corresponding `self.size`,
        the indicated dimension is reduced to 0.

        """
        width_post = min(size.width, self.size.width)
        height_post = min(size.height, self.size.height)

        size_post = Size2D(width=width_post, height=height_post)

        return self.from_centroid(centroid=self.centroid, size=self.size - size_post)

    def scaled(self, factor: Point2D) -> RectangleSubclass:
        """Scale rectangle by certain factor.

        The center point of the enlargement is the rectangle's centroid.

        """
        return self.from_centroid(centroid=self.centroid, size=self.size * factor)

    def invscaled(self, factor: Point2D) -> RectangleSubclass:
        """Scale rectangle by inversed factor."""
        return self.from_centroid(centroid=self.centroid, size=self.size / factor)

    def intersection(self, *others: Rectangle) -> RectangleSubclass | None:
        """Intersect many rectangles.

        Parameters
        ----------
        others : tuple of Rectangle objects
            `Rectangle` objects with which the rectangle is intersected.

        Returns
        -------
        RectangleSubclass
            A new object of the same class intersecting the rectangles.
            If the resulting intersection is empty, return `None`.

        """
        x0s, x1s = [self.x0], [self.x1]
        y0s, y1s = [self.y0], [self.y1]

        for other in others:
            x0s.append(other.x0)
            x1s.append(other.x1)
            y0s.append(other.y0)
            y1s.append(other.y1)

        x0, x1 = max(x0s), min(x1s)
        y0, y1 = max(y0s), min(y1s)

        if x0 > x1 or y0 > y1:
            return None

        return type(self)(p0=Point2D(x=x0, y=y0), p1=Point2D(x=x1, y=y1))

    def closure(self, *others: Rectangle) -> RectangleSubclass:
        """Enclose many rectangles with rectangle of minimal area.

        Parameters
        ----------
        others : tuple of Rectangle objects
            `Rectangle` objects with which the rectangle is enclosed.

        Returns
        -------
        RectangleSubclass
            A new object of the same class enclosing the rectangles.

        """
        x0s, x1s = [self.x0], [self.x1]
        y0s, y1s = [self.y0], [self.y1]

        for other in others:
            x0s.append(other.x0)
            x1s.append(other.x1)
            y0s.append(other.y0)
            y1s.append(other.y1)

        x0, x1 = min(x0s), max(x1s)
        y0, y1 = min(y0s), max(y1s)

        return type(self)(p0=Point2D(x=x0, y=y0), p1=Point2D(x=x1, y=y1))

    def iou(self, *others: Rectangle) -> list[confloat(ge=0.0, le=1.0)]:
        """Calculate intersection over union (IoU) of rectangles.

        Parameters
        ----------
        others : tuple of Rectangle objects
            `Rectangle` objects with which an intersection over union
            is measured.

        Returns
        -------
        result : list of floats in the range [0, 1]
            IoUs between the rectangle and other rectangles.

        """
        from respo.ml.core.processing.detection_utils import iou as iou_

        result = (
            iou_(
                x1=[self.x0, self.y0, self.x1, self.y1],
                x2=[[other.x0, other.y0, other.x1, other.y1] for other in others],
            )
            .flatten()
            .tolist()
        )

        return result

    def to_type(self, dtype: type, /) -> RectangleSubclass:
        """Cast fields to specified type."""
        rectangle = type(self).construct(
            p0=self.p0.to_type(dtype), p1=self.p1.to_type(dtype)
        )

        return rectangle

    def into_centroid(self) -> tuple[Point2D, Size2D]:
        """Convert rectangle to centroid representation.

        Representation includes a center point (centroid) and a size of
        a rectangle.

        Returns
        -------
        centroid : Point2D
            Central values along the X and Y axes.

        size : Size2D
            Sizes along the X and Y axes (width and height).

        """
        return self.centroid, self.size

    def into_min_corner(self) -> tuple[Point2D, Size2D]:
        """Convert rectangle to minimum corner representation.

        Representation includes a corner with minimum values and a size
        of a rectangle.

        Returns
        -------
        p0 : Point2D
            Minimum values along the X and Y axes.

        size : Size2D
            Sizes along the X and Y axes (width and height).

        """
        return self.p0, self.size

    def into_bitmask(self, size: Size2D) -> np.ndarray:
        """Convert rectangle to boolean mask of given size.

        Parameters
        ----------
        size : Size2D
            The shape of a boolean mask.

        Returns
        -------
        bitmask : 2D array of shape (height, width)
            A boolean mask.

        Notes
        -----
        Values that exceed the image are cropped to fit the image size.

        """
        bitmask = np.zeros((int(size.height), int(size.width)), dtype=bool)

        y0, x0 = max(0, round(self.y0)), max(0, round(self.x0))
        y1, x1 = max(0, round(self.y1) + 1), max(0, round(self.x1) + 1)

        bitmask[y0:y1, x0:x1] = True

        return bitmask

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Check if rectangle contains vector of points.

        Parameters
        ----------
        points : 2D array of shape (n_points, 2)
            Vector of points coordinates.

        Returns
        -------
        inside : 1D array of shape (n_points, )
            A boolean vector.

        """

        inside = (
            (points[:, 0] >= self.x0)
            & (points[:, 0] <= self.x1)
            & (points[:, 1] >= self.y0)
            & (points[:, 1] <= self.y1)
        )

        return inside

    @classmethod
    @validate_arguments
    def from_centroid(cls, centroid: Point2D, size: Size2D) -> RectangleSubclass:
        """Instantiate rectangle from centroid representation.

        Parameters
        ----------
        centroid : Point2D
            Central values along the X and Y axes.

        size : Size2D
            Sizes along the X and Y axes (width and height).

        Returns
        -------
        RectangleSubclass
            A new object of the same class.

        """
        offset = Point2D(x=size.width / 2, y=size.height / 2)

        return cls(p0=centroid - offset, p1=centroid + offset)

    @classmethod
    @validate_arguments
    def from_min_corner(cls, p0: Point2D, size: Size2D) -> RectangleSubclass:
        """Instantiate rectangle from minimum corner representation.

        Parameters
        ----------
        p0 : Point2D
            Minimum values along the X and Y axes.

        size : Size2D
            Sizes along the X and Y axes (width and height).

        Returns
        -------
        RectangleSubclass
            A new object of the same class.

        """
        p1 = Point2D(x=p0.x + size.width, y=p0.y + size.height)

        return cls(p0=p0, p1=p1)

    @classmethod
    def from_bitmask(cls, bitmask: npt.ArrayLike) -> RectangleSubclass:
        """Instantiate rectangle from boolean mask.

        Parameters
        ----------
        bitmask : 2D array of shape (height, width)
            A boolean mask of an axis-aligned rectangle.

        Returns
        -------
        RectangleSubclass
            A new object of the same class.

        Notes
        -----
        A rectangle is made up of points selected by taking the
        positions of the smallest and largest values along both axes.
        If a boolean mask is not representing a proper axis-aligned
        rectangle, then the resulting rectangle will be the closure of
        a boolean mask.

        """
        # bitmask = check_array(bitmask, n_dims=2, dtype=bool)

        rows, cols = np.argwhere(bitmask).T
        y0, y1 = np.quantile(rows, (0, 1))
        x0, x1 = np.quantile(cols, (0, 1))

        p0 = Point2D(x=x0, y=y0)
        p1 = Point2D(x=x1, y=y1)

        return cls(p0=p0, p1=p1)

    @classmethod
    def from_alignment(
        cls,
        alignment: Alignment,
        to: Rectangle,
        size: Size2D,
        border_margin: Size2D = _BORDER_MARGIN_DEFAULT,
    ) -> Rectangle:
        """Instantiate rectangle inner-aligned to other rectangle."""
        if alignment == Alignment.TOP:
            x = to.centroid.x - size.width // 2
            y = to.y0 + border_margin.height
        elif alignment == Alignment.BOTTOM:
            x = to.centroid.x - size.width // 2
            y = to.y1 - size.height - border_margin.height
        elif alignment == Alignment.LEFT:
            x = to.x0 + border_margin.width
            y = to.centroid.y - size.height // 2
        elif alignment == Alignment.RIGHT:
            x = to.x1 - size.width - border_margin.width
            y = to.centroid.y - size.height // 2
        elif alignment == Alignment.TOP_LEFT:
            x = to.x0 + border_margin.width
            y = to.y0 + border_margin.height
        elif alignment == Alignment.TOP_RIGHT:
            x = to.x1 - size.width - border_margin.width
            y = to.y0 + border_margin.height
        elif alignment == Alignment.BOTTOM_LEFT:
            x = to.x0 + border_margin.width
            y = to.y1 - size.height - border_margin.height
        elif alignment == Alignment.BOTTOM_RIGHT:
            x = to.x1 - size.width - border_margin.width
            y = to.y1 - size.height - border_margin.height
        elif alignment == Alignment.CENTER:
            x = to.centroid.x - size.width // 2
            y = to.centroid.y - size.height // 2
        else:
            raise ValueError("Unsupported alignment specified.")

        return cls.from_min_corner(p0=Point2D(x=x, y=y), size=size)

    @root_validator(pre=False, skip_on_failure=True)
    @classmethod
    def check_coordinates_compatibility(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check compatibility between corner points.

        The y0-x0 point coordinates should be lower than or equal
        to the y1-x1 point coordinates along both axes. Raise an
        error when this is violated.

        Paramters
        ---------
        values : dictionary
            A dictionary with the entire model's data.

        Returns
        -------
        values : dictionary
            Unchanged `values` input.

        Raises
        ------
        RectangleCornersError
            If corners are incompatible.

        """
        p0, p1 = values.get("p0"), values.get("p1")

        for coordinate in ["x", "y"]:
            lesser = getattr(p0, coordinate)
            greater = getattr(p1, coordinate)
            if lesser > greater:
                raise RectangleCornersError(
                    "Corners of the rectangle are incompatible. The p0 corner's "
                    f"{coordinate.upper()} coordinate must be lower than or equal to "
                    f"the p1 corner's {coordinate.upper()} coordinate, the violated "
                    f"condition is: {lesser} <= {greater}."
                )

        return values


class Ellipse(BaseModel, allow_inf_nan=False, frozen=True):
    """Ellipse representation on 2D plane.

    Parameters
    ----------
    center : Point 2D
        Ellipse center point.

    size : Size2D
        Length of the axes, where `width` corresponds to the major
        (horizontal if there is no rotation) axis and `height` to the
        minor (vertical if there is no rotation) axis.

    angle : float from [-np.pi, np.pi] range, optional (default=0.0)
        Rotation angle in radians from the Y axis to the ellipse's major
        axis.

    """

    center: Point2D
    size: Size2D
    angle: confloat(ge=-np.pi, le=np.pi) = 0.0

