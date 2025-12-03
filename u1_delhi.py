import processing
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, 
    QgsField, QgsWkbTypes, QgsSpatialIndex
)
from qgis.PyQt.QtCore import QVariant

# Configuration
ORIGINAL_LAYER_NAME = "building_delhi_bad"
SNAP_TOLERANCE = 1.0
GAP_MAX_AREA = 10.0

def get_layer(layer_name):
    layers = QgsProject.instance().mapLayersByName(layer_name)
    if not layers:
        raise Exception(f"Layer not found: {layer_name}")
    return layers[0]

def create_memory_layer(layer_crs, name, fields):
    layer = QgsVectorLayer(f"Polygon?crs={layer_crs.authid()}", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes(fields)
    layer.updateFields()
    return layer

def analyze_invalid_geometries(layer):
    invalid_geoms = []
    invalid_reasons = []
    feature_ids = []
    
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom and not geom.isNull() and not geom.isGeosValid():
            invalid_geoms.append(feature)
            error = geom.validateGeometry()
            invalid_reasons.append(error[0].what() if error else "Unknown error")
            feature_ids.append(feature.id())
    
    return invalid_geoms, invalid_reasons, feature_ids

def analyze_overlaps(layer):
    spatial_index = QgsSpatialIndex(layer.getFeatures())
    
    features_dict = {f.id(): f for f in layer.getFeatures()}
    overlaps = []
    overlap_count = 0
    total_overlap_area = 0
    checked_pairs = set()
    
    for fid, feat_a in features_dict.items():
        geom_a = feat_a.geometry()
        if not geom_a or geom_a.isNull():
            continue
        
        # Get only nearby features using spatial index
        bbox = geom_a.boundingBox()
        nearby_ids = spatial_index.intersects(bbox)
        
        for nearby_id in nearby_ids:
            # Skip self-comparison and already checked pairs
            if nearby_id <= fid:
                continue
            
            pair = (fid, nearby_id)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            feat_b = features_dict[nearby_id]
            geom_b = feat_b.geometry()
            if not geom_b or geom_b.isNull():
                continue
            
            if geom_a.intersects(geom_b):
                intersection = geom_a.intersection(geom_b)
                if (intersection and not intersection.isNull() and 
                    intersection.type() == QgsWkbTypes.PolygonGeometry):
                    overlap_area = intersection.area()
                    if overlap_area > 0.000001:
                        overlap_count += 1
                        total_overlap_area += overlap_area
                        overlap_feat = QgsFeature()
                        overlap_feat.setGeometry(intersection)
                        overlap_feat.setAttributes([fid, nearby_id, overlap_area])
                        overlaps.append(overlap_feat)
    
    return overlaps, overlap_count, total_overlap_area

def analyze_gaps(layer):
    gap_count = 0
    total_gap_area = 0
    gaps = []
    
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if not geom or geom.isNull():
            continue
            
        polygons = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
        
        for polygon in polygons:
            if len(polygon) > 1:
                for hole in polygon[1:]:
                    hole_geom = QgsGeometry.fromPolygonXY([hole])
                    hole_area = hole_geom.area()
                    gap_count += 1
                    total_gap_area += hole_area
                    gaps.append((feature.id(), hole_geom, hole_area))
    
    return gaps, gap_count, total_gap_area

def add_layers_to_project(layer_crs, invalid_data, overlap_data, gap_data, prefix):
    invalid_geoms, invalid_reasons, feature_ids = invalid_data
    overlaps, overlap_count, total_overlap_area = overlap_data
    gaps, gap_count, total_gap_area = gap_data
    
    if invalid_geoms:
        layer = create_memory_layer(
            layer_crs, 
            f"{prefix}_Invalid_Geometries",
            [QgsField("fid", QVariant.Int), QgsField("error_type", QVariant.String)]
        )
        features = []
        for i, feat in enumerate(invalid_geoms):
            new_feat = QgsFeature()
            new_feat.setGeometry(feat.geometry())
            new_feat.setAttributes([feature_ids[i], invalid_reasons[i]])
            features.append(new_feat)
        layer.dataProvider().addFeatures(features)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)
    
    if overlap_count > 0:
        layer = create_memory_layer(
            layer_crs,
            f"{prefix}_Overlapping_Areas",
            [QgsField("feature_a", QVariant.Int), 
             QgsField("feature_b", QVariant.Int),
             QgsField("overlap_area", QVariant.Double)]
        )
        layer.dataProvider().addFeatures(overlaps)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)
    
    if gap_count > 0:
        layer = create_memory_layer(
            layer_crs,
            f"{prefix}_Gaps_Holes",
            [QgsField("feature_id", QVariant.Int), QgsField("gap_area", QVariant.Double)]
        )
        features = []
        for fid, gap_geom, gap_area in gaps:
            feat = QgsFeature()
            feat.setGeometry(gap_geom)
            feat.setAttributes([fid, gap_area])
            features.append(feat)
        layer.dataProvider().addFeatures(features)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

def print_analysis_summary(prefix, feature_count, invalid_count, overlap_count, 
                          total_overlap_area, gap_count, total_gap_area):
    print(f"\n=== {prefix} ANALYSIS SUMMARY ===")
    print(f"Total Features: {feature_count}")
    print(f"Invalid Geometries: {invalid_count}")
    print(f"Overlap Count: {overlap_count}")
    print(f"Total Overlap Area: {total_overlap_area:.6f} sq units")
    print(f"Gap/Hole Count: {gap_count}")
    print(f"Total Gap Area: {total_gap_area:.6f} sq units")

# Main execution
original_layer = get_layer(ORIGINAL_LAYER_NAME)
layer_crs = original_layer.crs()

print("=== BEFORE ANALYSIS ===")
before_invalid = analyze_invalid_geometries(original_layer)
before_overlap = analyze_overlaps(original_layer)
before_gap = analyze_gaps(original_layer)

add_layers_to_project(layer_crs, before_invalid, before_overlap, before_gap, "BEFORE")
print_analysis_summary("BEFORE", original_layer.featureCount(), len(before_invalid[0]),
                       before_overlap[1], before_overlap[2], before_gap[1], before_gap[2])

# Geometry correction
print("\n=== GEOMETRY CORRECTION ===")
fixed_layer = processing.run("native:fixgeometries", {
    'INPUT': original_layer,
    'OUTPUT': 'memory:fixed_geometries'
})['OUTPUT']

corrected_layer = fixed_layer
if not layer_crs.isGeographic():
    corrected_layer = processing.run("native:snapgeometries", {
        'INPUT': fixed_layer,
        'REFERENCE_LAYER': fixed_layer,
        'TOLERANCE': SNAP_TOLERANCE,
        'BEHAVIOR': 1,
        'OUTPUT': 'memory:snapped_geometries'
    })['OUTPUT']
    
    corrected_layer = processing.run("native:deleteholes", {
        'INPUT': corrected_layer,
        'MIN_AREA': GAP_MAX_AREA,
        'OUTPUT': 'memory:corrected_geometries'
    })['OUTPUT']

corrected_layer.setName(f"{ORIGINAL_LAYER_NAME}_corrected")
QgsProject.instance().addMapLayer(corrected_layer)

print("\n=== AFTER ANALYSIS ===")
after_invalid = analyze_invalid_geometries(corrected_layer)
after_overlap = analyze_overlaps(corrected_layer)
after_gap = analyze_gaps(corrected_layer)

add_layers_to_project(layer_crs, after_invalid, after_overlap, after_gap, "AFTER")
print_analysis_summary("AFTER", corrected_layer.featureCount(), len(after_invalid[0]),
                      after_overlap[1], after_overlap[2], after_gap[1], after_gap[2])

# Final comparison
print("\n=== FINAL COMPARISON ===")
print(f"{'Metric':<30} | {'BEFORE':<12} | {'AFTER':<12}")
print("-" * 58)
print(f"{'Total Features':<30} | {original_layer.featureCount():<12} | {corrected_layer.featureCount():<12}")
print(f"{'Invalid Geometries':<30} | {len(before_invalid[0]):<12} | {len(after_invalid[0]):<12}")
print(f"{'Overlap Count':<30} | {before_overlap[1]:<12} | {after_overlap[1]:<12}")
print(f"{'Total Overlap Area':<30} | {before_overlap[2]:<12.4f} | {after_overlap[2]:<12.4f}")
print(f"{'Gap/Hole Count':<30} | {before_gap[1]:<12} | {after_gap[1]:<12}")
print(f"{'Total Gap Area':<30} | {before_gap[2]:<12.4f} | {after_gap[2]:<12.4f}")
print("\n=== COMPLETE ===")
