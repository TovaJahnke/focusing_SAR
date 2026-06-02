from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from sarpy.io.phase_history.cphd import CPHDDetails, CPHDReader


def _signal_raw_dtype(cphd_details: CPHDDetails) -> np.dtype:
    data = cphd_details.cphd_meta.Data
    if data.SignalCompressionID is not None:
        raise ValueError("Compressed CPHD signal fallback is not supported")
    if data.SignalArrayFormat == "CF8":
        return np.dtype(">f4")
    if data.SignalArrayFormat == "CI4":
        return np.dtype(">i2")
    if data.SignalArrayFormat == "CI2":
        return np.dtype(">i1")
    raise ValueError(f"Unsupported CPHD signal array format: {data.SignalArrayFormat}")


def _full_slice(the_range: Union[None, int, slice], limit: int) -> Union[int, slice]:
    if the_range is None:
        return slice(0, limit, 1)
    if isinstance(the_range, slice):
        return the_range
    idx = int(the_range)
    if idx < 0:
        idx += limit
    if not (0 <= idx < limit):
        raise IndexError(f"Index {the_range} out of bounds for length {limit}")
    return idx


def _channel_identifier(cphd_details: CPHDDetails, index: Union[int, str]) -> str:
    channels = cphd_details.cphd_meta.Data.Channels
    if isinstance(index, str):
        for entry in channels:
            if entry.Identifier == index:
                return index
        raise KeyError(f"CPHD channel identifier '{index}' not found")
    int_index = int(index)
    if not (0 <= int_index < len(channels)):
        raise ValueError(f"CPHD channel index {int_index} outside valid range [0, {len(channels)})")
    return channels[int_index].Identifier


def _needs_truncation_fallback(cphd_details: CPHDDetails) -> bool:
    file_size = Path(cphd_details.file_name).stat().st_size
    raw_dtype = _signal_raw_dtype(cphd_details)
    block_offset = int(cphd_details.cphd_header.SIGNAL_BLOCK_BYTE_OFFSET)
    for entry in cphd_details.cphd_meta.Data.Channels:
        signal_offset = block_offset + int(entry.SignalArrayByteOffset)
        expected_bytes = int(entry.NumVectors) * int(entry.NumSamples) * 2 * raw_dtype.itemsize
        if signal_offset + expected_bytes > file_size:
            return True
    return False


class TruncatedCPHDReader:
    def __init__(self, cphd_details: CPHDDetails):
        self._cphd_details = cphd_details
        self._cphd_meta = cphd_details.cphd_meta
        self._cphd_header = cphd_details.cphd_header
        self._signal_memmap = {}
        self._pvp_memmap = {}
        self.warnings: list[str] = []

        raw_dtype = _signal_raw_dtype(cphd_details)
        pvp_dtype = self._cphd_meta.PVP.get_vector_dtype()
        file_size = Path(cphd_details.file_name).stat().st_size
        signal_block_offset = int(self._cphd_header.SIGNAL_BLOCK_BYTE_OFFSET)
        pvp_block_offset = int(self._cphd_header.PVP_BLOCK_BYTE_OFFSET)

        for entry in self._cphd_meta.Data.Channels:
            identifier = entry.Identifier
            advertised_vectors = int(entry.NumVectors)
            num_samples = int(entry.NumSamples)
            bytes_per_vector = num_samples * 2 * raw_dtype.itemsize
            signal_offset = signal_block_offset + int(entry.SignalArrayByteOffset)
            available_bytes = max(0, file_size - signal_offset)
            available_vectors = min(advertised_vectors, available_bytes // bytes_per_vector)
            if available_vectors <= 0:
                raise ValueError(
                    f"CPHD signal for channel {identifier} starts at byte {signal_offset}, "
                    f"but the file only has {available_bytes} bytes available"
                )
            if available_vectors < advertised_vectors:
                self.warnings.append(
                    f"Channel {identifier} advertises {advertised_vectors} vectors, "
                    f"but only {available_vectors} complete vectors fit in the file; "
                    f"processing will use the intact prefix only"
                )
                entry.NumVectors = available_vectors

            self._signal_memmap[identifier] = np.memmap(
                cphd_details.file_name,
                dtype=raw_dtype,
                mode="r",
                offset=signal_offset,
                shape=(int(entry.NumVectors), num_samples, 2),
            )

            pvp_offset = pvp_block_offset + int(entry.PVPArrayByteOffset)
            self._pvp_memmap[identifier] = np.memmap(
                cphd_details.file_name,
                dtype=pvp_dtype,
                mode="r",
                offset=pvp_offset,
                shape=(int(entry.NumVectors),),
            )

    @property
    def cphd_meta(self):
        return self._cphd_meta

    @property
    def cphd_header(self):
        return self._cphd_header

    @property
    def cphd_details(self):
        return self._cphd_details

    def read_raw(self, *ranges, index: Union[int, str] = 0, squeeze: bool = True) -> np.ndarray:
        identifier = _channel_identifier(self._cphd_details, index)
        memmap = self._signal_memmap[identifier]
        full_ranges = list(ranges[: memmap.ndim])
        while len(full_ranges) < memmap.ndim:
            full_ranges.append(None)
        subscript = tuple(_full_slice(rng, memmap.shape[axis]) for axis, rng in enumerate(full_ranges))
        out = np.array(memmap[subscript], copy=True)
        return np.squeeze(out) if squeeze else out

    def read_pvp_variable(
        self,
        variable: str,
        index: Union[int, str],
        the_range: Union[None, int, slice] = None,
    ) -> np.ndarray | None:
        identifier = _channel_identifier(self._cphd_details, index)
        memmap = self._pvp_memmap[identifier]
        if variable not in memmap.dtype.fields:
            return None
        subscript = _full_slice(the_range, memmap.shape[0])
        return np.array(memmap[variable][subscript], copy=True)

    def close(self) -> None:
        self._signal_memmap.clear()
        self._pvp_memmap.clear()
        self._cphd_details.close()


def open_cphd_reader(cphd_path: Union[str, Path]):
    cphd_details = CPHDDetails(str(Path(cphd_path).expanduser().resolve()))
    if not _needs_truncation_fallback(cphd_details):
        return CPHDReader(cphd_details)

    reader = TruncatedCPHDReader(cphd_details)
    for warning in reader.warnings:
        print(f"[warn] {warning}")
    return reader
