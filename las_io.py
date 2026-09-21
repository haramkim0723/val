"""LAS 파일 헤더와 XYZ 좌표를 numpy로 읽는 최소 리더 (laspy 불필요).

LAS 1.0 ~ 1.4의 비압축 .las만 지원한다. .laz(압축)는 지원하지 않는다.
4천만 점 이상도 다룰 수 있도록 memmap + 청크 단위로 읽는다.
"""

import os
import struct
from dataclasses import dataclass

import numpy as np

CHUNK = 4_000_000


@dataclass
class LasInfo:
    path: str
    version: tuple
    point_format: int
    record_length: int
    offset_to_points: int
    num_points: int
    scale: tuple
    offset: tuple


def read_header(path):
    with open(path, "rb") as f:
        h = f.read(375)
    if len(h) < 227 or h[:4] != b"LASF":
        raise ValueError(f"LAS 파일이 아닙니다: {path}")

    major, minor = h[24], h[25]
    (offset_to_points,) = struct.unpack_from("<I", h, 96)
    format_byte = h[104]
    if format_byte & 0xC0:
        raise ValueError(f".laz(압축) 파일은 지원하지 않습니다. .las로 변환하세요: {path}")
    (record_length,) = struct.unpack_from("<H", h, 105)
    (num_points,) = struct.unpack_from("<I", h, 107)
    if minor >= 4:
        (num_points,) = struct.unpack_from("<Q", h, 247)

    info = LasInfo(
        path=path,
        version=(major, minor),
        point_format=format_byte & 0x3F,
        record_length=record_length,
        offset_to_points=offset_to_points,
        num_points=num_points,
        scale=struct.unpack_from("<3d", h, 131),
        offset=struct.unpack_from("<3d", h, 155),
    )
    if offset_to_points + num_points * record_length > os.path.getsize(path):
        raise ValueError(f"헤더의 점 개수에 비해 파일이 너무 작습니다(잘린 파일?): {path}")
    return info


def iter_xyz_chunks(info, chunk=CHUNK):
    """(x, y, z) 정수 좌표(int32)를 청크 단위로 반환. 실제 좌표 = 정수 * scale + offset."""
    if info.num_points == 0:
        return
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<i4", "<i4", "<i4"],
            "offsets": [0, 4, 8],
            "itemsize": info.record_length,
        }
    )
    points = np.memmap(
        info.path,
        dtype=dtype,
        mode="r",
        offset=info.offset_to_points,
        shape=(info.num_points,),
    )
    for start in range(0, info.num_points, chunk):
        part = points[start : start + chunk]
        yield part["x"], part["y"], part["z"]


def to_ref_grid(raw, axis, info, ref):
    """`info` 파일의 정수 좌표를 `ref` 파일의 정수 격자 좌표(int64)로 변환.

    두 파일의 scale/offset이 같으면 그대로, 다르면 실제 좌표를 거쳐 반올림한다.
    """
    if info.scale[axis] == ref.scale[axis] and info.offset[axis] == ref.offset[axis]:
        return raw.astype(np.int64)
    real = raw.astype(np.float64) * info.scale[axis] + info.offset[axis]
    return np.rint((real - ref.offset[axis]) / ref.scale[axis]).astype(np.int64)
