"""Layout primitives shared by all layout engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):  # numpy scalars would break JSON serialisation
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in (self.x0, self.y0, self.x1, self.y1))

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersect(self, other: Box) -> Box:
        return Box(max(self.x0, other.x0), max(self.y0, other.y0),
                   min(self.x1, other.x1), min(self.y1, other.y1))

    def union(self, other: Box) -> Box:
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1))

    def overlap_ratio(self, other: Box) -> float:
        """Share of *this* box covered by `other`."""
        return self.intersect(other).area / self.area if self.area else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, v: Iterable[float]) -> Box:
        return cls(*[float(x) for x in v])


@dataclass
class Region:
    box: Box
    kind: str  # see config.RegionKind
    score: float = 1.0
    source: str = ""  # engine + original label, for debugging
    id: int = -1
    group: int | None = None  # sidebar/frame id; members are read as one unit
    order: int | None = None
    dropped: str | None = None  # reason, when filtered out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = self.box.to_list()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        d = dict(d)
        d["box"] = Box.from_list(d["box"])
        return cls(**d)


@dataclass
class PageLayout:
    page: int
    side: str
    width: int
    height: int
    content: Box  # page minus configured margins
    regions: list[Region] = field(default_factory=list)
    frames: list[Box] = field(default_factory=list)  # sidebar boxes; Region.group indexes this

    def kept(self) -> list[Region]:
        return sorted((r for r in self.regions if r.dropped is None),
                      key=lambda r: (r.order if r.order is not None else 1e9))

    def to_dict(self) -> dict:
        return {
            "page": self.page, "side": self.side, "width": self.width, "height": self.height,
            "content": self.content.to_list(),
            "frames": [f.to_list() for f in self.frames],
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PageLayout:
        return cls(
            page=d["page"], side=d["side"], width=d["width"], height=d["height"],
            content=Box.from_list(d["content"]),
            frames=[Box.from_list(f) for f in d.get("frames", [])],
            regions=[Region.from_dict(r) for r in d["regions"]],
        )


def merge_overlapping(boxes: list[Box], pad: float = 0.0) -> list[Box]:
    """Union boxes that overlap (optionally after padding) until stable."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        for b in boxes:
            for i, o in enumerate(out):
                grown = Box(o.x0 - pad, o.y0 - pad, o.x1 + pad, o.y1 + pad)
                if grown.intersect(b).area > 0:
                    out[i] = o.union(b)
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes
