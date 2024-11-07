import pdal
import numpy as np
import json
import laspy
import pandas as pd
import geopandas as gpd

# Merge point clouds
def merge_point_clouds():
    merge_cmd = [
        "pdal", "merge",
        "sbet_h1_corrected.las", "sbet_h2_corrected.las", "sbet_h3_corrected.las",
        "sbet_h4_corrected.las", "sbet_h5_corrected.las", "sbet_v1_corrected.las",
        "sbet_v2_corrected.las", "sbet_v3_corrected.las", "sbet_v4_corrected.las",
        "sbet_v5_corrected.las", "sbet_v6_corrected.las",
        "--writers.las.minor_version=4", "--writers.las.extra_dims=all",
        "--writers.las.a_srs=EPSG:32613", "corrected_intensity.las"
    ]
    pdal.Pipeline(" ".join(merge_cmd)).execute()

# Read and process new las file
def read_and_process():
    pipeline = pdal.Reader.las(filename='corrected_intensity.las').pipeline()
    pipeline.execute()
    arr_gs = pipeline.arrays[0].copy()

    # Remove outliers beyond 3 SD of median
    pipeline = pdal.Filter.mad(dimension='CorrIntens', k=3.0).pipeline(arr_gs)
    pipeline.execute()
    arr_gs = pipeline.arrays[0].copy()

    # Normalize
    max_intens = np.max(arr_gs['CorrIntens'])
    arr_gs['CorrIntens'] = arr_gs['CorrIntens'] / max_intens

    # Output TIFF
    write_tiff(arr_gs, 'CorrIntens', 'corrected_intensity.tif')

    # Scale and add reflectance dimension
    scale_factor = -0.03
    arr_gs['Refl'] = arr_gs['CorrIntens'] + scale_factor

    # Filter reflectance values to only include snow surfaces
    pipeline = pdal.Filter.range(limits='Refl[0.40:]').pipeline(arr_gs)
    pipeline.execute()
    arr_gs = pipeline.arrays[0].copy()

    # Convert incidence angles to degrees
    arr_gs['Incidence'] = np.rad2deg(arr_gs['Incidence'])

    # Export as las file
    write_las(arr_gs, 'reflectance.las')

    return arr_gs

def write_las(arr, filename):
    pipeline = pdal.Writer.las(
        minor_version=4,
        extra_dims='all',
        a_srs='EPSG:32613',
        filename=filename
    ).pipeline(arr)
    pipeline.execute()

# Read lookup table and assign grain size
def assign_grain_size():
    lookup_table = pd.read_csv('brf_lidar_1064_2.csv', index_col='grain_size')
    las = laspy.read('reflectance.las')
    
    grain_sizes = []
    for i in range(len(las.points)):
        incidence = las.Incidence[i]
        corr_intens = las.Refl[i]
        
        closest_inc_col = min(lookup_table.columns, key=lambda col: abs(float(col.split('_')[1]) - incidence))
        closest_grain_size = lookup_table[closest_inc_col].sub(corr_intens).abs().idxmin()
        
        grain_sizes.append(closest_grain_size)

    las.GrainSize = grain_sizes
    las.write('grain_size.las')

# Clip grain_size.las to SBB boundary
def clip_to_boundary():
    gdf = gpd.read_file('SBB_basin_poly_ASO3m.shp')
    polygon_wkt = gdf.geometry[0].wkt

    clip_to_boundary = {
        "pipeline": [
            {"type": "readers.las", "filename": "grain_size.las"},
            {"type": "filters.crop", "polygon": polygon_wkt},
            {
                "type": "writers.las",
                "filename": "grain_size_clip.las",
                "minor_version": "4",
                "extra_dims": "all",
                "a_srs": "EPSG:32613"
            }
        ]
    }

    with open('clip_to_boundary.json', 'w') as f:
        json.dump(clip_to_boundary, f)

    pdal.Pipeline('clip_to_boundary.json').execute()

# Export TIFFs
def write_tiff(arr, dimension, filename, srs='EPSG:32613', output_type='idw', resolution='1.0'):
    pipeline = pdal.Writer.gdal(
        dimension=dimension,
        output_type=output_type,
        resolution=resolution,
        override_srs=srs,
        filename=filename
    ).pipeline(arr)
    pipeline.execute()

def main():
    merge_point_clouds()
    arr_gs = read_and_process()
    assign_grain_size()
    clip_to_boundary()

    # Output TIFFs for various dimensions
    dimensions = {
        'Incidence': 'incidence.tif',
        'CorrIntens': 'corrected_intensity.tif',
        'Refl': 'reflectance.tif',
        'GrainSize': 'grain_size.tif'
    }
    for dimension, output_file in dimensions.items():
        write_tiff(arr_gs, dimension, output_file)

if __name__ == "__main__":
    main()
