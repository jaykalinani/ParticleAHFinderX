#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert ParticleAHFinderX particle snapshots to connected VTK surfaces.

The production input is openPMD/ADIOS2 BP5. Per-rank debug TSV is also
accepted so topology reconstruction can be tested without an openPMD Python
installation. Reconstruction uses only stable surface IDs and immutable
logical angular indices; AMReX particle or MPI-rank order is irrelevant.
"""

import argparse
import collections
import glob
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET


TSV_FIELDS = (
    "x", "y", "z", "particleId", "cpu", "surfaceId", "generation",
    "angularLevel", "angularLevels", "itheta", "iphi", "ntheta", "nphi",
    "lifecycle", "role",
    "validity", "sampledLevel", "height", "relaxationVelocity",
    "sampledLevelChanged", "expansion", "normalX", "normalY", "normalZ",
    "areaWeight")

INTEGER_FIELDS = {
    "particleId", "cpu", "surfaceId", "generation", "angularLevel",
    "angularLevels", "itheta", "iphi", "ntheta", "nphi", "lifecycle",
    "role", "validity", "sampledLevel",
    "sampledLevelChanged"}

POINT_FIELDS = (
    "particleId", "cpu", "surfaceId", "generation", "angularLevel",
    "angularLevels", "itheta", "iphi", "ntheta", "nphi", "lifecycle",
    "role", "validity", "sampledLevel",
    "sampledLevelChanged", "height", "relaxationVelocity", "expansion",
    "normalX", "normalY", "normalZ", "areaWeight", "isSyntheticPole")

VTK_TYPES = {
    "particleId": "Int64",
    "cpu": "Int32",
    "surfaceId": "Int64",
    "generation": "Int32",
    "angularLevel": "Int32",
    "angularLevels": "Int32",
    "itheta": "Int32",
    "iphi": "Int32",
    "ntheta": "Int32",
    "nphi": "Int32",
    "lifecycle": "Int32",
    "role": "Int32",
    "validity": "Int32",
    "sampledLevel": "Int32",
    "sampledLevelChanged": "Int32",
    "height": "Float64",
    "relaxationVelocity": "Float64",
    "expansion": "Float64",
    "normalX": "Float64",
    "normalY": "Float64",
    "normalZ": "Float64",
    "areaWeight": "Float64",
    "isSyntheticPole": "UInt8",
}

ITERATION_PATTERN = re.compile(r"\.it([0-9]{8})\.")


class ConversionError(RuntimeError):
    """An input snapshot cannot form the documented logical surface."""


class Frame:
    def __init__(self, iteration, time, columns):
        self.iteration = int(iteration)
        self.time = float(time)
        self.columns = columns


def iteration_from_name(path):
    match = ITERATION_PATTERN.search(path.name)
    if match is None:
        raise ConversionError(
            "Input name does not contain '.it########.': {}".format(path))
    return int(match.group(1))


def expand_inputs(inputs):
    paths = []
    for value in inputs:
        matches = [pathlib.Path(path) for path in glob.glob(value)]
        if not matches and pathlib.Path(value).exists():
            matches = [pathlib.Path(value)]
        if not matches:
            raise ConversionError("No input matches '{}'".format(value))
        for path in matches:
            if path.is_dir() and path.suffix != ".bp5":
                bp5 = sorted(path.glob("particleahfinderx_particles.it*.bp5"))
                tsv = sorted(path.glob(
                    "particleahfinderx_particles.it*.rank*.tsv"))
                paths.extend(bp5)
                paths.extend(tsv)
            else:
                paths.append(path)
    unique = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ConversionError("No ParticleAHFinderX snapshots were found")
    return unique


def empty_columns():
    return {name: [] for name in TSV_FIELDS}


def read_tsv(paths):
    grouped = collections.defaultdict(list)
    for path in paths:
        grouped[iteration_from_name(path)].append(path)

    frames = []
    for iteration, shards in sorted(grouped.items()):
        columns = empty_columns()
        for path in sorted(shards):
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip() or line.startswith("#"):
                        continue
                    values = line.split()
                    if len(values) != len(TSV_FIELDS):
                        raise ConversionError(
                            "{}:{} has {} columns; expected {}".format(
                                path, line_number, len(values),
                                len(TSV_FIELDS)))
                    for name, value in zip(TSV_FIELDS, values):
                        converter = int if name in INTEGER_FIELDS else float
                        columns[name].append(converter(value))
        frames.append(Frame(iteration, iteration, columns))
    return frames


def read_bp5(paths):
    try:
        import numpy
        import openpmd_api as io
    except ImportError as error:
        raise ConversionError(
            "BP5 conversion requires the openPMD-api Python module; "
            "install openpmd-api with ADIOS2 support or use the bounded TSV "
            "input for a debug conversion") from error

    def scalar_component(species, name):
        record_name = "id" if name == "particleId" else name
        return species[record_name][io.Record_Component.SCALAR]

    def component(species, name):
        if name in ("normalX", "normalY", "normalZ"):
            return species["normal"][name[-1].lower()]
        return scalar_component(species, name)

    frames = []
    seen_iterations = set()
    for path in sorted(paths):
        series = io.Series(str(path), io.Access.read_only)
        try:
            for iteration_index in series.iterations:
                iteration_index = int(iteration_index)
                if iteration_index in seen_iterations:
                    raise ConversionError(
                        "Duplicate BP5 iteration {}".format(iteration_index))
                seen_iterations.add(iteration_index)
                iteration = series.iterations[iteration_index]
                columns = empty_columns()

                for species_name in sorted(iteration.particles):
                    species = iteration.particles[species_name]
                    components = {
                        "x": species["position"]["x"],
                        "y": species["position"]["y"],
                        "z": species["position"]["z"],
                    }
                    for name in TSV_FIELDS[3:]:
                        components[name] = component(species, name)

                    pending = {
                        name: component.load_chunk()
                        for name, component in components.items()
                    }
                    offsets = {}
                    if "positionOffset" in species:
                        for axis in ("x", "y", "z"):
                            offsets[axis] = species["positionOffset"][axis].load_chunk()
                    series.flush()

                    for name, values in pending.items():
                        array = numpy.asarray(values)
                        if name in INTEGER_FIELDS:
                            columns[name].extend(int(value) for value in array)
                        else:
                            columns[name].extend(float(value) for value in array)
                    for axis, values in offsets.items():
                        offset = numpy.asarray(values)
                        start = len(columns[axis]) - len(offset)
                        for i, value in enumerate(offset):
                            columns[axis][start + i] += float(value)

                try:
                    time = float(iteration.get_attribute("cactusTime"))
                except Exception:
                    time = float(iteration_index)
                frames.append(Frame(iteration_index, time, columns))
        finally:
            series.close()
    return sorted(frames, key=lambda frame: frame.iteration)


def constant_value(columns, indices, name, surface_key):
    values = {columns[name][index] for index in indices}
    if len(values) != 1:
        raise ConversionError(
            "Surface {} has inconsistent {} values: {}".format(
                surface_key, name, sorted(values)))
    return values.pop()


def ring_mean(columns, logical, name, itheta, nphi):
    return sum(columns[name][logical[(itheta, iphi)]]
               for iphi in range(nphi)) / nphi


def pole_float(columns, logical, name, north, ntheta, nphi):
    if north:
        inner = ring_mean(columns, logical, name, 0, nphi)
        next_inner = ring_mean(columns, logical, name, 1, nphi)
    else:
        inner = ring_mean(columns, logical, name, ntheta - 1, nphi)
        next_inner = ring_mean(columns, logical, name, ntheta - 2, nphi)
    return (9.0 * inner - next_inner) / 8.0


def reconstruct_frame(frame):
    columns = frame.columns
    counts = {name: len(values) for name, values in columns.items()}
    if not counts or len(set(counts.values())) != 1:
        raise ConversionError(
            "Iteration {} has inconsistent record lengths: {}".format(
                frame.iteration, counts))
    particle_count = next(iter(counts.values()))
    if particle_count == 0:
        raise ConversionError(
            "Iteration {} contains no particles".format(frame.iteration))

    grouped = collections.defaultdict(list)
    for index in range(particle_count):
        key = (columns["surfaceId"][index], columns["generation"][index],
               columns["angularLevel"][index])
        grouped[key].append(index)

    points = []
    point_data = {name: [] for name in POINT_FIELDS}
    triangles = []

    for surface_key, indices in sorted(grouped.items()):
        ntheta = int(constant_value(
            columns, indices, "ntheta", surface_key))
        nphi = int(constant_value(columns, indices, "nphi", surface_key))
        if ntheta < 2 or nphi != 2 * ntheta:
            raise ConversionError(
                "Surface {} has unsupported dimensions {}x{}; expected "
                "ntheta >= 2 and nphi == 2*ntheta".format(
                    surface_key, ntheta, nphi))
        if len(indices) != ntheta * nphi:
            raise ConversionError(
                "Surface {} contains {} points; expected {}".format(
                    surface_key, len(indices), ntheta * nphi))

        logical = {}
        for index in indices:
            key = (int(columns["itheta"][index]),
                   int(columns["iphi"][index]))
            if key in logical:
                raise ConversionError(
                    "Surface {} repeats logical point {}".format(
                        surface_key, key))
            if not (0 <= key[0] < ntheta and 0 <= key[1] < nphi):
                raise ConversionError(
                    "Surface {} has out-of-range logical point {}".format(
                        surface_key, key))
            logical[key] = index
        expected = {(itheta, iphi) for itheta in range(ntheta)
                    for iphi in range(nphi)}
        missing = expected.difference(logical)
        if missing:
            raise ConversionError(
                "Surface {} is missing logical point {}".format(
                    surface_key, min(missing)))

        base = len(points)
        center = [0.0, 0.0, 0.0]
        for itheta in range(ntheta):
            theta = math.pi * (itheta + 0.5) / ntheta
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            for iphi in range(nphi):
                phi = 2.0 * math.pi * (iphi + 0.5) / nphi
                direction = (sin_theta * math.cos(phi),
                             sin_theta * math.sin(phi), cos_theta)
                index = logical[(itheta, iphi)]
                height = columns["height"][index]
                for axis, name in enumerate(("x", "y", "z")):
                    center[axis] += columns[name][index] - height * direction[axis]
        center = [value / (ntheta * nphi) for value in center]

        for itheta in range(ntheta):
            for iphi in range(nphi):
                index = logical[(itheta, iphi)]
                points.append(tuple(columns[name][index]
                                    for name in ("x", "y", "z")))
                for name in POINT_FIELDS[:-1]:
                    point_data[name].append(columns[name][index])
                point_data["isSyntheticPole"].append(0)

        pole_indices = []
        for north in (True, False):
            height = pole_float(
                columns, logical, "height", north, ntheta, nphi)
            points.append((center[0], center[1],
                           center[2] + (height if north else -height)))
            pole_indices.append(len(points) - 1)

            point_data["particleId"].append(-1 if north else -2)
            point_data["cpu"].append(-1)
            point_data["surfaceId"].append(surface_key[0])
            point_data["generation"].append(surface_key[1])
            point_data["angularLevel"].append(surface_key[2])
            point_data["angularLevels"].append(
                int(constant_value(columns, indices, "angularLevels",
                                   surface_key)))
            point_data["itheta"].append(-1 if north else ntheta)
            point_data["iphi"].append(-1)
            point_data["ntheta"].append(ntheta)
            point_data["nphi"].append(nphi)
            for name in ("lifecycle", "role"):
                point_data[name].append(
                    int(constant_value(columns, indices, name, surface_key)))
            point_data["validity"].append(
                max(int(columns["validity"][index]) for index in indices))
            point_data["sampledLevel"].append(
                max(int(columns["sampledLevel"][index])
                    for index in indices))
            point_data["sampledLevelChanged"].append(
                max(int(columns["sampledLevelChanged"][index])
                    for index in indices))
            point_data["height"].append(height)
            for name in ("relaxationVelocity", "expansion", "normalX",
                         "normalY", "normalZ", "areaWeight"):
                point_data[name].append(pole_float(
                    columns, logical, name, north, ntheta, nphi))
            point_data["isSyntheticPole"].append(1)

        north, south = pole_indices
        point = lambda itheta, iphi: base + itheta * nphi + iphi % nphi
        for iphi in range(nphi):
            triangles.append((north, point(0, iphi), point(0, iphi + 1)))
        for itheta in range(ntheta - 1):
            for iphi in range(nphi):
                upper_left = point(itheta, iphi)
                lower_left = point(itheta + 1, iphi)
                lower_right = point(itheta + 1, iphi + 1)
                upper_right = point(itheta, iphi + 1)
                triangles.append((upper_left, lower_left, lower_right))
                triangles.append((upper_left, lower_right, upper_right))
        for iphi in range(nphi):
            triangles.append(
                (point(ntheta - 1, iphi), south,
                 point(ntheta - 1, iphi + 1)))

    return points, point_data, triangles


def values_text(values, floating=False):
    if floating:
        return " ".join("{:.17g}".format(float(value)) for value in values)
    return " ".join(str(int(value)) for value in values)


def add_data_array(parent, name, vtk_type, values, components=None):
    attributes = {"type": vtk_type, "Name": name, "format": "ascii"}
    if components is not None:
        attributes["NumberOfComponents"] = str(components)
    array = ET.SubElement(parent, "DataArray", attributes)
    array.text = values_text(values, vtk_type.startswith("Float"))


def indent_xml(element, level=0):
    """Backport ElementTree.indent for the Python 3.6 analysis environment."""
    whitespace = "\n" + level * "  "
    child_whitespace = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = whitespace
    if level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def write_vtp(path, frame, points, point_data, triangles):
    vtk = ET.Element("VTKFile", {
        "type": "PolyData", "version": "1.0", "byte_order": "LittleEndian"})
    polydata = ET.SubElement(vtk, "PolyData")
    field_data = ET.SubElement(polydata, "FieldData")
    add_data_array(field_data, "cactusIteration", "Int64",
                   [frame.iteration])
    add_data_array(field_data, "cactusTime", "Float64", [frame.time])
    piece = ET.SubElement(polydata, "Piece", {
        "NumberOfPoints": str(len(points)),
        "NumberOfVerts": "0",
        "NumberOfLines": "0",
        "NumberOfStrips": "0",
        "NumberOfPolys": str(len(triangles)),
    })
    vtk_points = ET.SubElement(piece, "Points")
    add_data_array(vtk_points, "Position", "Float64",
                   [coordinate for point in points for coordinate in point], 3)

    vtk_point_data = ET.SubElement(piece, "PointData", {"Scalars": "expansion"})
    for name in POINT_FIELDS:
        add_data_array(vtk_point_data, name, VTK_TYPES[name], point_data[name])

    polys = ET.SubElement(piece, "Polys")
    add_data_array(polys, "connectivity", "Int64",
                   [vertex for triangle in triangles for vertex in triangle])
    add_data_array(polys, "offsets", "Int64",
                   [3 * (index + 1) for index in range(len(triangles))])

    indent_xml(vtk)
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def write_pvd(path, entries):
    vtk = ET.Element("VTKFile", {
        "type": "Collection", "version": "1.0",
        "byte_order": "LittleEndian"})
    collection = ET.SubElement(vtk, "Collection")
    for time, filename in entries:
        ET.SubElement(collection, "DataSet", {
            "timestep": "{:.17g}".format(time),
            "group": "", "part": "0", "file": filename})
    indent_xml(vtk)
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+",
        help="BP5 snapshots, per-rank TSV shards, globs, or an output directory")
    parser.add_argument(
        "--output-dir", default="particleahfinderx-vtk",
        help="directory for VTP frames and the PVD index")
    parser.add_argument(
        "--prefix", default="particleahfinderx_surfaces",
        help="base name for generated VTP/PVD files")
    parser.add_argument(
        "--input-format", choices=("auto", "bp5", "tsv"), default="auto",
        help="choose BP5 or debug TSV when an output directory contains both")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    try:
        inputs = expand_inputs(arguments.inputs)
        bp5 = [path for path in inputs if path.suffix == ".bp5"]
        tsv = [path for path in inputs if path.suffix == ".tsv"]
        if arguments.input_format == "bp5":
            tsv = []
        elif arguments.input_format == "tsv":
            bp5 = []
        elif bp5:
            tsv = []
        if not bp5 and not tsv:
            raise ConversionError(
                "No {} inputs were found".format(arguments.input_format))
        frames = read_bp5(bp5) if bp5 else read_tsv(tsv)

        output_dir = pathlib.Path(arguments.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for frame in frames:
            points, point_data, triangles = reconstruct_frame(frame)
            filename = "{}.it{:08d}.vtp".format(
                arguments.prefix, frame.iteration)
            write_vtp(output_dir / filename, frame, points, point_data, triangles)
            entries.append((frame.time, filename))
            print("Wrote {}: {} points, {} triangles".format(
                output_dir / filename, len(points), len(triangles)))
        pvd = output_dir / "{}.pvd".format(arguments.prefix)
        write_pvd(pvd, entries)
        print("Wrote {}".format(pvd))
    except (ConversionError, OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
