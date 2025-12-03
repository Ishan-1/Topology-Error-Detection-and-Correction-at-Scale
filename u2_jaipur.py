import processing
from qgis.core import (
    QgsVectorLayer, QgsProject, QgsGeometry, QgsPointXY, QgsSpatialIndex,
    QgsFeature, QgsField, QgsVectorFileWriter
)
from PyQt5.QtCore import QVariant
from collections import defaultdict
import math


# PARAMS FOR ALGO
road_layer_name = "Reprojected"

snap_tolerance = 0.5
break_tolerance = 0.5

dangling_threshold = 10
short_segment_threshold = 5
long_edge_threshold = 50.0
collinearity_tol_deg = 5.0


def build_vertex_connections(layer, precision=10):
    """
    Build vertex connections graph and cache feature geometries.
    Returns: (vertex_connections, feature_cache, vertex_to_coords)
    """
    connections = defaultdict(set)
    feature_cache = {}
    vertex_to_coords = {}
    
    for feat in layer.getFeatures():
        fid = feat.id()
        geom = feat.geometry()
        if not geom or geom.isEmpty():
            continue
        
    
        vertices = list(geom.vertices())
        if len(vertices) < 2:
            continue
        
        # Cache the line coordinates
        if geom.isMultipart():
            lines = geom.asMultiPolyline()
            feature_cache[fid] = lines
        else:
            line = geom.asPolyline()
            feature_cache[fid] = [line]
        
        # Process endpoints only
        start = vertices[0]
        end = vertices[-1]
        
        start_key = (round(start.x(), precision), round(start.y(), precision))
        end_key = (round(end.x(), precision), round(end.y(), precision))
        
        connections[start_key].add(fid)
        connections[end_key].add(fid)
        
        vertex_to_coords[start_key] = QgsPointXY(start.x(), start.y())
        vertex_to_coords[end_key] = QgsPointXY(end.x(), end.y())
    
    return connections, feature_cache, vertex_to_coords


class UnionFind:
    """Efficient Union-Find data structure for connected components"""
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.component_size = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.component_size[x] = 1
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        self.component_size[px] += self.component_size[py]
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_components_info(self):
        """Returns (num_components, component_sizes_list)"""
        roots = set()
        for node in self.parent:
            roots.add(self.find(node))
        
        sizes = [self.component_size[root] for root in roots]
        return len(roots), sizes


def count_connected_components_fast(vertex_connections, all_feature_ids):
    uf = UnionFind()
    for fid in all_feature_ids:
        uf.find(fid)  
    for coord, fids in vertex_connections.items():
        fids_list = list(fids)
        if len(fids_list) > 1:
            first = fids_list[0]
            for fid in fids_list[1:]:
                uf.union(first, fid)
    
    return uf.get_components_info()


def get_angle_fast(coords1, coords2, common_coord_key, line1_is_start, line2_is_start):
    try:
        if line1_is_start:
            p_common_1 = coords1[0]
            p_neighbor_1 = coords1[1] if len(coords1) > 1 else coords1[0]
        else:
            p_common_1 = coords1[-1]
            p_neighbor_1 = coords1[-2] if len(coords1) > 1 else coords1[-1]

        if line2_is_start:
            p_neighbor_2 = coords2[1] if len(coords2) > 1 else coords2[0]
        else:
            p_neighbor_2 = coords2[-2] if len(coords2) > 1 else coords2[-1]
        
        p_node = QgsPointXY(p_common_1.x(), p_common_1.y())
        p1 = QgsPointXY(p_neighbor_1.x(), p_neighbor_1.y())
        p2 = QgsPointXY(p_neighbor_2.x(), p_neighbor_2.y())
        
        angle1 = p_node.azimuth(p1)
        angle2 = p_node.azimuth(p2)
        
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
            
        return diff
    except Exception as e:
        return 180 

original_layer = QgsProject.instance().mapLayersByName(road_layer_name)
if not original_layer:
    raise Exception(f"Layer not found: {road_layer_name}")
original_layer = original_layer[0]

layer_crs = original_layer.crs()
is_geographic = layer_crs.isGeographic()

print(f"\n{'='*80}")
print(f"DETECTION AND CORRECTION PIPELINE (OPTIMIZED)")
print(f"{'='*80}")
print(f"Layer: {road_layer_name}")
print(f"CRS: {layer_crs.authid()} ({'Geographic' if is_geographic else 'Projected'})")
print(f"Total features: {original_layer.featureCount()}")

if is_geographic:
    print("\nWARNING: Geographic CRS detected - adjusting thresholds")
    METER = 0.000009
    snap_tolerance = 0.5 * METER
    break_tolerance = 0.5 * METER
    dangling_threshold = 10 * METER
    short_segment_threshold = 5 * METER
    long_edge_threshold = 50 * METER
print(f"\n{'='*80}")
print("PHASE 1: DIAGNOSING TOPOLOGY ISSUES")
print(f"{'='*80}")


print("Building connectivity graph...")
vertex_connections, feature_cache, vertex_to_coords = build_vertex_connections(original_layer, precision=10)

degree_1 = [v for v, fids in vertex_connections.items() if len(fids) == 1]
degree_2 = [v for v, fids in vertex_connections.items() if len(fids) == 2]
degree_3_plus = [v for v, fids in vertex_connections.items() if len(fids) >= 3]

print(f"\nNetwork topology:")
print(f"  Dangles (degree 1):        {len(degree_1)}")
print(f"  Pseudo-nodes (degree 2):   {len(degree_2)}")
print(f"  Intersections (degree 3+): {len(degree_3_plus)}")

dangle_types = {
    'micro_gap': 0,
    'near_intersection': 0,
    'short_stub': 0,
    'medium_stub': 0,
    'long_deadend': 0,
    'isolated': 0
}


print("Building spatial index...")
spatial_idx = QgsSpatialIndex(original_layer.getFeatures())
# Detect dangles
analyze_sample_size = min(len(degree_1), 1000)
sample_dangles = degree_1[:analyze_sample_size]
min_distances = []


feature_lengths = {}
for feat in original_layer.getFeatures():
    feature_lengths[feat.id()] = feat.geometry().length()

for dangle_vertex in sample_dangles:
    dangle_point = vertex_to_coords[dangle_vertex]
    dangle_geom = QgsGeometry.fromPointXY(dangle_point)
    
    dangle_fid = next(iter(vertex_connections[dangle_vertex]))
    segment_length = feature_lengths[dangle_fid]
    
    search_radius = 50 if not is_geographic else 0.0005
    nearby_fids = spatial_idx.intersects(dangle_geom.buffer(search_radius, 5).boundingBox())
    
    min_dist = float('inf')
    for nearby_fid in nearby_fids:
        if nearby_fid == dangle_fid:
            continue
        nearby_feat = original_layer.getFeature(nearby_fid)
        dist = dangle_geom.distance(nearby_feat.geometry())
        if dist < min_dist:
            min_dist = dist
    
    if min_dist != float('inf'):
        min_distances.append(min_dist)
    
    # Classify dangle
    if min_dist < 1 or (is_geographic and min_dist < 0.00001):
        dangle_types['micro_gap'] += 1
    elif min_dist < 2 or (is_geographic and min_dist < 0.00002):
        dangle_types['near_intersection'] += 1
    elif segment_length < 5 or (is_geographic and segment_length < 0.00005):
        dangle_types['short_stub'] += 1
    elif segment_length < 20 or (is_geographic and segment_length < 0.0002):
        dangle_types['medium_stub'] += 1
    elif min_dist > 50 or (is_geographic and min_dist > 0.0005):
        dangle_types['isolated'] += 1
    else:
        dangle_types['long_deadend'] += 1

print(f"\nDangle classification (sample: {len(sample_dangles)}):")
for dtype, count in dangle_types.items():
    pct = (count / len(sample_dangles)) * 100 if sample_dangles else 0
    print(f"  {dtype:20s}: {count:3d} ({pct:5.1f}%)")

micro_gap_pct = (dangle_types['micro_gap'] / len(sample_dangles) * 100) if sample_dangles else 0
if micro_gap_pct > 30:
    print(f"\nPRIMARY ISSUE: MICRO-GAPS - Increasing snap tolerance")
    snap_tolerance *= 2
    break_tolerance *= 2

print(f"\n{'='*80}")
print("PHASE 2: APPLYING CORRECTIONS")
print(f"{'='*80}")

# Step 1: Snap and break
print(f"\n1. Running snap + break (tolerance: {snap_tolerance})...")
try:
    snapped = processing.run(
        "grass7:v.clean",
        {
            'input': original_layer,
            'type': [1],
            'tool': [5],
            'threshold': snap_tolerance,
            '-b': False,
            '-c': True,
            'output': 'memory:Snapped',
            'error': 'memory:SnapErrors'
        }
    )
    working_layer = snapped['output']
    if isinstance(working_layer, str):
        working_layer = QgsVectorLayer(working_layer, "Snapped", "ogr")
    
    broken = processing.run(
        "grass7:v.clean",
        {
            'input': working_layer,
            'type': [1],
            'tool': [0],
            'threshold': break_tolerance,
            '-b': False,
            '-c': False,
            'output': 'memory:Broken',
            'error': 'memory:BreakErrors'
        }
    )
    working_layer = broken['output']
    if isinstance(working_layer, str):
        working_layer = QgsVectorLayer(working_layer, "Broken", "ogr")
    
    final_snap = processing.run(
        "grass7:v.clean",
        {
            'input': working_layer,
            'type': [1],
            'tool': [5],
            'threshold': snap_tolerance,
            '-b': False,
            '-c': False,
            'output': 'memory:FinalSnap',
            'error': 'memory:FinalSnapErrors'
        }
    )
    working_layer = final_snap['output']
    if isinstance(working_layer, str):
        working_layer = QgsVectorLayer(working_layer, "FinalSnap", "ogr")
    
    print(f"   Complete: {working_layer.featureCount()} features")
    
except Exception as e:
    print(f"   GRASS unavailable, trying native tools...")
    try:
        snapped = processing.run(
            "native:snapgeometries",
            {
                'INPUT': original_layer,
                'REFERENCE_LAYER': original_layer,
                'TOLERANCE': snap_tolerance,
                'BEHAVIOR': 3,
                'OUTPUT': 'memory:Snapped'
            }
        )
        working_layer = snapped['OUTPUT']
        
        split = processing.run(
            "native:splitwithlines",
            {
                'INPUT': working_layer,
                'LINES': working_layer,
                'OUTPUT': 'memory:Split'
            }
        )
        working_layer = split['OUTPUT']
        print(f"   Complete: {working_layer.featureCount()} features")
    except Exception as e2:
        print(f"   Native tools failed, using original layer")
        working_layer = original_layer

# Remove short dangles 
print(f"\n2. Removing short dangles (<{dangling_threshold})...")

# Rebuild connections with working layer
vertex_connections_work, feature_cache_work, _ = build_vertex_connections(working_layer, precision=10)

feature_lengths_work = {}
for feat in working_layer.getFeatures():
    feature_lengths_work[feat.id()] = feat.geometry().length()

candidate_remove = set()
for coord, fids in vertex_connections_work.items():
    if len(fids) == 1:
        fid = next(iter(fids))
        if feature_lengths_work.get(fid, float('inf')) <= dangling_threshold:
            candidate_remove.add(fid)

n_remove = len(candidate_remove)
removal_pct = (n_remove / working_layer.featureCount()) * 100

if n_remove > 0 and removal_pct < 10:
    ids_string = ','.join(str(i) for i in candidate_remove)
    cleaned = processing.run(
        "native:extractbyexpression",
        {
            'INPUT': working_layer,
            'EXPRESSION': f'$id NOT IN ({ids_string})',
            'OUTPUT': 'memory:Cleaned'
        }
    )['OUTPUT']
    working_layer = cleaned
    print(f"   Removed {n_remove} short dangles ({removal_pct:.2f}%)")
else:
    print(f"   Skipping (removal rate: {removal_pct:.2f}%)")

print(f"\n3. Fixing pseudo-nodes...")
vertex_connections_pseudo, feature_cache_pseudo, _ = build_vertex_connections(working_layer, precision=8)

coord_to_lines = defaultdict(list)
for coord, fids in vertex_connections_pseudo.items():
    for fid in fids:
        lines = feature_cache_pseudo[fid]
        for line in lines:
            if len(line) >= 2:
                start_key = (round(line[0].x(), 8), round(line[0].y(), 8))
                end_key = (round(line[-1].x(), 8), round(line[-1].y(), 8))
                if start_key == coord:
                    coord_to_lines[coord].append((fid, True, False))
                if end_key == coord:
                    coord_to_lines[coord].append((fid, False, True))

pseudo_nodes = [coord for coord, fids in vertex_connections_pseudo.items() if len(fids) == 2]
print(f"   Found {len(pseudo_nodes)} pseudo-nodes")

line_features = {feat.id(): feat for feat in working_layer.getFeatures()}

# Classify pseudo-nodes with cached geometries
line_ids_to_merge = set()

for coord_key in pseudo_nodes:
    connected = coord_to_lines.get(coord_key, [])
    
    if len(connected) == 2:
        line1_id, line1_is_start, _ = connected[0]
        line2_id, line2_is_start, _ = connected[1]
        
        line1 = line_features[line1_id]
        line2 = line_features[line2_id]
        
        len1 = line1.geometry().length()
        len2 = line2.geometry().length()
        min_adj_len = min(len1, len2)
        
        # Get cached coordinates
        coords1 = feature_cache_pseudo[line1_id][0]
        coords2 = feature_cache_pseudo[line2_id][0]
        
        turn_angle = get_angle_fast(coords1, coords2, coord_key, line1_is_start, line2_is_start)
        
        is_error = False
        
        # Rule 3: Exactly one short segment
        if (len1 < short_segment_threshold and len2 >= short_segment_threshold) or \
           (len2 < short_segment_threshold and len1 >= short_segment_threshold):
            is_error = True
        
        # Rule 5: Non-error - intentional nodes on straight roads
        elif (min_adj_len > long_edge_threshold) and (turn_angle <= collinearity_tol_deg):
            is_error = False
        
        # Both segments short or nearly collinear
        elif (len1 < short_segment_threshold * 3 and len2 < short_segment_threshold * 3):
            is_error = True
        elif (turn_angle <= collinearity_tol_deg * 2 and max(len1, len2) < long_edge_threshold * 0.5):
            is_error = True
        
        if is_error:
            line_ids_to_merge.add(line1_id)
            line_ids_to_merge.add(line2_id)

print(f"   Identified {len(line_ids_to_merge)} segments to merge")

if line_ids_to_merge:
    ids_string = ','.join(str(id) for id in line_ids_to_merge)
    
    valid_segments = processing.run(
        "native:extractbyexpression", {
            'INPUT': working_layer,
            'EXPRESSION': f'$id NOT IN ({ids_string})',
            'OUTPUT': 'memory:Valid_Segments'
        }
    )['OUTPUT']
    
    error_segments = processing.run(
        "native:extractbyexpression", {
            'INPUT': working_layer,
            'EXPRESSION': f'$id IN ({ids_string})',
            'OUTPUT': 'memory:Error_Segments'
        }
    )['OUTPUT']
    
    field_names = [field.name() for field in error_segments.fields()]
    dissolve_fields = []
    if 'name' in field_names:
        dissolve_fields.append('name')
    if 'highway' in field_names:
        dissolve_fields.append('highway')
    
    dissolved_errors = processing.run(
        "native:dissolve", {
            'INPUT': error_segments,
            'FIELD': dissolve_fields,
            'OUTPUT': 'memory:Dissolved_Errors'
        }
    )['OUTPUT']
    
    working_layer = processing.run(
        "native:mergevectorlayers", {
            'LAYERS': [valid_segments, dissolved_errors],
            'OUTPUT': 'memory:Final'
        }
    )['OUTPUT']
    
    print(f"   Merged pseudo-node errors")
else:
    print(f"   No pseudo-node errors to correct")


working_layer.setName("Topology_Fixed")
QgsProject.instance().addMapLayer(working_layer)
print("\nCalculating network connectivity metrics...")


original_fids = [f.id() for f in original_layer.getFeatures()]
original_components, original_sizes = count_connected_components_fast(vertex_connections, original_fids)
original_largest = max(original_sizes) if original_sizes else 0
original_connectivity = (original_largest / original_layer.featureCount() * 100) if original_layer.featureCount() > 0 else 0

# Rebuild connections for final layer
final_vertex_connections, _, _ = build_vertex_connections(working_layer, precision=10)

final_count = working_layer.featureCount()
final_dangles = len([v for v, fids in final_vertex_connections.items() if len(fids) == 1])
final_pseudo = len([v for v, fids in final_vertex_connections.items() if len(fids) == 2])

final_fids = [f.id() for f in working_layer.getFeatures()]
final_components, final_sizes = count_connected_components_fast(final_vertex_connections, final_fids)
final_largest = max(final_sizes) if final_sizes else 0
final_connectivity = (final_largest / final_count * 100) if final_count > 0 else 0

# Calculate metrics
component_reduction = original_components - final_components
connectivity_improvement = final_connectivity - original_connectivity

print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")
print(f"Segments:      {original_layer.featureCount()} → {final_count} ({final_count - original_layer.featureCount():+d})")
print(f"Dangles:       {len(degree_1)} → {final_dangles} ({len(degree_1) - final_dangles} fixed, {((len(degree_1) - final_dangles) / len(degree_1) * 100):.1f}%)")
print(f"Pseudo-nodes:  {len(degree_2)} → {final_pseudo} ({len(degree_2) - final_pseudo} fixed, {((len(degree_2) - final_pseudo) / len(degree_2) * 100):.1f}%)")
print(f"\n{'='*80}")
print("NETWORK CONNECTIVITY ANALYSIS")
print(f"{'='*80}")
print(f"Connected components:     {original_components} → {final_components} ({component_reduction:+d})")
print(f"Largest component size:   {original_largest} → {final_largest} segments")
print(f"Network connectivity:     {original_connectivity:.1f}% → {final_connectivity:.1f}% ({connectivity_improvement:+.1f}%)")
print(f"{'='*80}")