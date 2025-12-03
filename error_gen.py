

from pathlib import Path
import argparse
import json
import math
import random

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input vector layer (GeoJSON/Shapefile)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--epsg", type=int, default=32643, help="Projected EPSG for metric operations (default: 32643 - UTM zone for Jaipur)")
    p.add_argument("--seed", type=int, default=0)

    # Gap parameters
    p.add_argument("--gap-prob", type=float, default=0.1, help="Per-feature probability to apply a gap operation")
    p.add_argument("--gap-min", type=float, default=0.1, help="Minimum gap distance in metres")
    p.add_argument("--gap-max", type=float, default=0.7, help="Maximum gap distance in metres")
    p.add_argument("--gap-method", choices=["shrink","translate"], default="shrink", help="How to create gaps")

    # Overlap parameters
    p.add_argument("--overlap-prob", type=float, default=0.05, help="Per-feature probability to apply an overlap operation")
    p.add_argument("--overlap-min", type=float, default=0.1, help="Minimum overlap translation (m)")
    p.add_argument("--overlap-max", type=float, default=1.0, help="Maximum overlap translation (m)")
    p.add_argument("--overlap-method", choices=["translate","buffer"], default="translate")

    # Filters
    p.add_argument("--min-area", type=float, default=5.0, help="Ignore polygons smaller than this area (m^2)")

    return p.parse_args()


def ensure_projected(gdf, epsg_target):
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS. Reproject to an appropriate projected CRS first or let the script assume EPSG via --epsg.")
    if gdf.crs.to_epsg() != epsg_target:
        return gdf.to_crs(epsg_target)
    return gdf


def build_spatial_index(gdf):
    # geopandas exposes sindex
    return gdf.sindex


def pick_candidate_neighbors(gdf, sindex, idx, max_search=5):
    # Return list of neighbor indices ordered by distance (approx via bbox)
    geom = gdf.geometry.iloc[idx]
    possible = list(sindex.intersection(geom.bounds))
    possible = [i for i in possible if i != idx]
    # compute actual distances and sort
    dists = []
    for i in possible:
        d = geom.distance(gdf.geometry.iloc[i])
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    return [i for _, i in dists[:max_search]]


def shrink_polygon(poly, d):
    # negative buffer inward; return original if operation invalid
    p2 = poly.buffer(-d)
    if p2.is_empty:
        return poly
    # keep polygon or multipolygon
    return p2


def translate_along_vector(poly, vx, vy, d):
    # translate by d along (vx,vy)
    # normalize
    norm = math.hypot(vx, vy)
    if norm == 0:
        return poly
    ux, uy = vx / norm, vy / norm
    return translate(poly, xoff=ux * d, yoff=uy * d)


def ensure_valid(geom):
    if geom.is_valid:
        return geom
    # best-effort repair
    g = geom.buffer(0)
    if g.is_valid:
        return g
    # fallback to original
    return geom


def polygon_aspect_ratio(poly):
    # approximate width/height of minimum rotated rectangle
    mrr = poly.minimum_rotated_rectangle
    if not isinstance(mrr, Polygon):
        return 0
    coords = list(mrr.exterior.coords)
    # rectangle has 5 pts (last==first)
    edge_lengths = [math.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1]) for i in range(4)]
    edge_lengths.sort()
    if edge_lengths[0] == 0:
        return 0
    return edge_lengths[3] / edge_lengths[0]


def generate_corruptions(gdf, epsg_target, args):
    random.seed(args.seed)
    sindex = build_spatial_index(gdf)

    corrupted_geoms = []
    metadata = []

    for idx, row in gdf.iterrows():
        orig = row.geometry
        geom = orig
        if orig.area < args.min_area:
            corrupted_geoms.append(geom)
            metadata.append({"orig_idx": int(idx), "applied": []})
            continue

        applied = []

        # GAP
        if random.random() < args.gap_prob:
            d = random.uniform(args.gap_min, args.gap_max)
            if args.gap_method == "shrink":
                new_geom = shrink_polygon(geom, d)
                if new_geom.equals(geom):
                    # fallback: small translate
                    neighs = pick_candidate_neighbors(gdf, sindex, idx, max_search=1)
                    if neighs:
                        nidx = neighs[0]
                        ngeom = gdf.geometry.iloc[nidx]
                        vx = geom.centroid.x - ngeom.centroid.x
                        vy = geom.centroid.y - ngeom.centroid.y
                        new_geom = translate_along_vector(geom, vx, vy, d)
                new_geom = ensure_valid(new_geom)
                applied.append({"type": "gap_shrink", "d": d})
                geom = new_geom
            else:  # translate method
                neighs = pick_candidate_neighbors(gdf, sindex, idx, max_search=1)
                if neighs:
                    nidx = neighs[0]
                    ngeom = gdf.geometry.iloc[nidx]
                    vx = geom.centroid.x - ngeom.centroid.x
                    vy = geom.centroid.y - ngeom.centroid.y
                else:
                    # random direction if no neighbor
                    theta = random.random() * 2 * math.pi
                    vx, vy = math.cos(theta), math.sin(theta)
                new_geom = translate_along_vector(geom, vx, vy, d)
                new_geom = ensure_valid(new_geom)
                applied.append({"type": "gap_translate", "d": d})
                geom = new_geom

        # OVERLAP
        if random.random() < args.overlap_prob:
            d = random.uniform(args.overlap_min, args.overlap_max)
            if args.overlap_method == "translate":
                # pick nearest neighbor and translate towards it
                neighs = pick_candidate_neighbors(gdf, sindex, idx, max_search=1)
                if neighs:
                    nidx = neighs[0]
                    ngeom = gdf.geometry.iloc[nidx]
                    vx = ngeom.centroid.x - geom.centroid.x
                    vy = ngeom.centroid.y - geom.centroid.y
                else:
                    theta = random.random() * 2 * math.pi
                    vx, vy = math.cos(theta), math.sin(theta)
                new_geom = translate_along_vector(geom, vx, vy, d)
                new_geom = ensure_valid(new_geom)
                applied.append({"type": "overlap_translate", "d": d})
                geom = new_geom
            else:  # buffer method
                out = geom.buffer(d)
                # intersect with nearest neighbors to create overlap regions
                neighs = pick_candidate_neighbors(gdf, sindex, idx, max_search=5)
                overlapped_with = []
                for nidx in neighs:
                    inter = out.intersection(gdf.geometry.iloc[nidx])
                    if not inter.is_empty:
                        overlapped_with.append(int(nidx))
                if overlapped_with:
                    # union out with original to create overlap
                    new_geom = geom.union(out)
                    new_geom = ensure_valid(new_geom)
                    applied.append({"type": "overlap_buffer", "d": d, "neighs": overlapped_with})
                    geom = new_geom

        corrupted_geoms.append(geom)
        metadata.append({"orig_idx": int(idx), "applied": applied})

    # build corrupted gdf
    corrupted = gdf.copy()
    corrupted["geometry"] = corrupted_geoms
    return corrupted, metadata


def extract_gap_and_overlap_polygons(orig_gdf, corrupted_gdf, min_area=0.01, aspect_ratio_min=2.5):
    # For each feature, compute difference areas that indicate gaps or overlaps.
    gap_parts = []
    overlap_parts = []

    for o, c in zip(orig_gdf.geometry, corrupted_gdf.geometry):
        # gap: part in original but not in corrupted (i.e., removed area)
        gap = o.difference(c)
        if not gap.is_empty:
            geoms = getattr(gap, 'geoms', [gap])
            for g in geoms:
                if g.area >= min_area and polygon_aspect_ratio(g) >= aspect_ratio_min:
                    gap_parts.append(g)
        # overlap: part in corrupted but not in original (i.e., new overlapped area)
        ov = c.difference(o)
        if not ov.is_empty:
            geoms = getattr(ov, 'geoms', [ov])
            for g in geoms:
                if g.area >= min_area and polygon_aspect_ratio(g) >= aspect_ratio_min:
                    overlap_parts.append(g)

    gaps_gdf = gpd.GeoDataFrame(geometry=gap_parts, crs=orig_gdf.crs)
    overlaps_gdf = gpd.GeoDataFrame(geometry=overlap_parts, crs=orig_gdf.crs)
    return gaps_gdf, overlaps_gdf


def save_outputs(corrupted_gdf, gaps_gdf, overlaps_gdf, metadata, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    corrupted_path = out_dir / "corrupted_buildings.geojson"
    gaps_path = out_dir / "gaps.geojson"
    overlaps_path = out_dir / "overlaps.geojson"
    meta_path = out_dir / "metadata.json"

    corrupted_gdf.to_file(corrupted_path, driver="GeoJSON")
    if len(gaps_gdf):
        gaps_gdf.to_file(gaps_path, driver="GeoJSON")
    else:
        # create empty file
        gpd.GeoDataFrame(geometry=[], crs=corrupted_gdf.crs).to_file(gaps_path, driver="GeoJSON")
    if len(overlaps_gdf):
        overlaps_gdf.to_file(overlaps_path, driver="GeoJSON")
    else:
        gpd.GeoDataFrame(geometry=[], crs=corrupted_gdf.crs).to_file(overlaps_path, driver="GeoJSON")

    with open(meta_path, "w") as fh:
        json.dump({"metadata": metadata}, fh, indent=2)

    print(f"Saved: {corrupted_path}\n       {gaps_path}\n       {overlaps_path}\n       {meta_path}")


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    gdf = gpd.read_file(in_path)

    # If input CRS is not set, assume latlon and reproject to epsg
    if gdf.crs is None:
        # try to guess it's EPSG:4326
        gdf.set_crs(4326, inplace=True)

    gdf = ensure_projected(gdf, args.epsg)

    corrupted_gdf, metadata = generate_corruptions(gdf, args.epsg, args)

    # extract sliver/overlap polygons
    gaps_gdf, overlaps_gdf = extract_gap_and_overlap_polygons(gdf, corrupted_gdf)

    save_outputs(corrupted_gdf, gaps_gdf, overlaps_gdf, metadata, out_dir)


if __name__ == "__main__":
    main()
