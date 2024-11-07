import argparse
import pdal
import numpy as np
import glob
import os
import json
import laspy
import logging
import pandas as pd
import geopandas as gpd
from numba import njit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LiDAR data with configurable parameters.")
    
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the directory containing LiDAR data.")
    parser.add_argument("--input_laz_file", type=str, required=True,
                        help="Name of the input .laz file to process.")
    parser.add_argument("--epsg_code", type=str, required=True,
                        help="EPSG code for the output spatial reference system (e.g., 'EPSG:32613').")
    parser.add_argument("--trajectory_csv", type=str, required=True,
                        help="Path to the trajectory CSV file.")
    parser.add_argument("--reference_range", type=int, default=1339,
                        help="Reference range distance for intensity correction.")
    parser.add_argument("--min_scan_angle", type=int, default=-15,
                        help="Minimum scan angle to filter points.")
    parser.add_argument("--max_scan_angle", type=int, default=15,
                        help="Maximum scan angle to filter points.")
    
    return parser.parse_args()


#Error handling
def handle_file_errors(func):
    """
    Decorator to handle errors and logging for file operations.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper

@handle_file_errors
def read_las_file(input_laz_file):
    pipeline = pdal.Reader.las(filename=input_laz_file).pipeline()
    pipeline.execute()
    return pipeline.arrays[0].copy()

class UnsupportedFilterTypeError(Exception):
    """Exception raised for unsupported filter types."""
    pass

#Apply filters; find ground points
def apply_filter(arr, filter_type, **kwargs):
    """
    Applies a specified PDAL filter to a point cloud array.

    :param arr: The input point cloud array.
    :param filter_type: The type of PDAL filter to apply.
    :param kwargs: Additional keyword arguments for the filter.
    :return: The filtered point cloud array, or None if an error occurs or the result is empty.
    """
    try:
        if filter_type == 'range':
            pipeline = pdal.Filter.range(**kwargs).pipeline(arr)
        elif filter_type == 'returns':
            pipeline = pdal.Filter.returns(**kwargs).pipeline(arr)
        elif filter_type == 'elm':
            pipeline = pdal.Filter.elm().pipeline(arr)
        elif filter_type == 'smrf':
            pipeline = pdal.Filter.smrf(**kwargs).pipeline(arr)
        elif filter_type == 'voxelcenternearestneighbor':
            pipeline = pdal.Filter.voxelcenternearestneighbor(**kwargs).pipeline(arr)
        elif filter_type == 'ferry':
            pipeline = pdal.Filter.ferry(**kwargs).pipeline(arr)
        else:
            raise UnsupportedFilterTypeError(f"Unsupported filter type: {filter_type}")

        pipeline.execute()
        filtered_array = pipeline.arrays[0].copy()
        if len(filtered_array) == 0:
            logging.warning(f"Point cloud is empty after applying {filter_type} filter")
            return None
        return filtered_array
    except Exception as e:
        logging.error(f"Error applying {filter_type} filter: {e}")
        return None

#Filter GPS time for analysis per flight line
def filter_by_gps_time(arr, start_time, end_time):
    if start_time >= end_time:
        logging.error(f"Invalid time range: start_time ({start_time}) must be less than end_time ({end_time})")
        return None
    if start_time < 0 or end_time < 0:
        logging.error(f"Invalid time range: start_time ({start_time}) and end_time ({end_time}) must be non-negative")
        return None

    min_gps_time = np.min(arr['GpsTime'])
    max_gps_time = np.max(arr['GpsTime'])
    if start_time < min_gps_time or end_time > max_gps_time:
        logging.error(f"Time range [{start_time}, {end_time}] is outside the point cloud's GPS time range [{min_gps_time}, {max_gps_time}]")
        return None

    return apply_filter(arr, 'range', limits=f'GpsTime[{start_time}:{end_time}]')

def filter_single_returns(arr):
    return apply_filter(arr, 'returns', groups='only')

def filter_scan_angle(arr, min_angle, max_angle):
    return apply_filter(arr, 'range', limits=f'ScanAngleRank[{min_angle}:{max_angle}]')

def remove_noise_and_find_ground(arr):
    arr = apply_filter(arr, 'elm')
    return apply_filter(arr, 'smrf', slope='0.2', window='16', threshold='0.45', scalar='1.2')

def thin_point_cloud(arr, cell_size):
    return apply_filter(arr, 'voxelcenternearestneighbor', cell=cell_size)

def add_dimensions(arr):
    dimensions = '=>Range, =>Incidence, =>CorrIntens, =>Refl, =>GrainSize'
    return apply_filter(arr, 'ferry', dimensions=dimensions)

@handle_file_errors
def write_las_file(arr, filename, epsg_code):
    pipeline = pdal.Writer.las(
        minor_version=4,
        extra_dims='all',
        a_srs=epsg_code,
        filename=filename
    ).pipeline(arr)
    pipeline.execute()

def process_point_cloud(name, start_time, end_time, input_laz_file, epsg_code, min_scan_angle, max_scan_angle):
    arr = read_las_file(input_laz_file)
    arr = filter_by_gps_time(arr, start_time, end_time)
    arr = filter_single_returns(arr)
    arr = filter_scan_angle(arr, min_scan_angle, max_scan_angle)
    arr = remove_noise_and_find_ground(arr)
    arr = thin_point_cloud(arr, '1.0')
    arr = add_dimensions(arr)
    gps_time_filename = f'{name}.las'
    write_las_file(arr, gps_time_filename, epsg_code)

def process_all_point_clouds(gps_times_with_names, input_laz_file, epsg_code, file_path, min_scan_angle, max_scan_angle):
    input_laz_file = os.path.join(file_path, input_laz_file)
    for item in gps_times_with_names:
        process_point_cloud(
            name=item["name"],
            start_time=item["start"],
            end_time=item["end"],
            input_laz_file=input_laz_file,
            epsg_code=epsg_code,
            min_scan_angle=min_scan_angle,
            max_scan_angle=max_scan_angle
        )

def process_trajectory(file_path, trajectory_csv):
    flight_line_las = glob.glob(os.path.join(file_path, '*.las'))
    output_trajectory_files = []
    if not flight_line_las:
        logging.error("No LAS files found in the specified directory.")
        return output_trajectory_files
    try:
        trajectory_data = np.loadtxt(trajectory_csv, skiprows=1, delimiter=',')
    except Exception as e:
        logging.error(f"Error reading trajectory file {trajectory_csv}: {e}")
        return output_trajectory_files

    for las_f_name in flight_line_las:
        with open(las_f_name, "rb+") as f:
            f.seek(6)
            f.write(bytes([17, 0, 0, 0]))

        pipeline = pdal.Reader.las(filename=las_f_name).pipeline() 
        pipeline.execute()
        arr = pipeline.arrays[0]

        # Filter the trajectory data based on the current LAS file's GPS time range
        filtered_trajectory_data = trajectory_data[(trajectory_data[:,0] > arr['GpsTime'].min()) & (trajectory_data[:,0] < arr['GpsTime'].max())]

        num_points = len(arr['GpsTime'])
        num_times = len(filtered_trajectory_data[:,0])

        output_trajectory_file = las_f_name[:-4] + '_trajectory.txt'
        output_trajectory_files.append(output_trajectory_file)
        with open(output_trajectory_file, 'w') as f:
            np.savetxt(f, filtered_trajectory_data, header='GPSTime Easting Northing Elevation Roll Pitch Yaw', comments='')

    return output_trajectory_files
    
            
@njit(fastmath=True)
def normalize_intensity(gps_times, trajectory_times):
    """Find the nearest index in trajectory_times for each time in gps_times."""
    for t in gps_times:
        i = np.searchsorted(trajectory_times, t)
        if i == 0 or (i < len(trajectory_times) and np.abs(t - trajectory_times[i-1]) < np.abs(t - trajectory_times[i])):
            if i - 1 >= 0:
                yield i - 1
            else:
                yield i
        else:
            if i < len(trajectory_times):
                yield i
            else:
                yield i - 1

def cos_incidence_angle(X, Y, Z, n1, n2, n3):
    """Calculate the cosine of the incidence angle."""
    numerator = (-X * n1) + (-Y * n2) + (-Z * n3)
    denominator = np.sqrt(X**2 + Y**2 + Z**2) * np.sqrt(n1**2 + n2**2 + n3**2)
    return numerator / denominator

def correct_intensity(raw_intensity, range_dist, reference_range, incidence):
    """Correct intensity based on range distance and incidence angle."""
    return (raw_intensity * np.square(range_dist)) / (np.square(reference_range) * np.cos(incidence))

def read_trajectory_data(traj_file_name):
    """Read and return the trajectory data from a CSV file."""
    try:
        return np.loadtxt(traj_file_name, skiprows=1, delimiter=' ')
    except Exception as e:
        logging.error(f"Error reading trajectory file {traj_file_name}: {e}")
        return None

def calculate_range_and_incidence(arr, trajectory_data, indices):
    """Calculate the range and incidence angle for the point cloud."""
    arr = filter_normal(arr, knn=8)
    X, Y, Z = arr['X'] - trajectory_data[indices,1], arr['Y'] - trajectory_data[indices,2], arr['Z'] - trajectory_data[indices,3]
    R = np.sqrt(X**2 + Y**2 + Z**2)
    theta = cos_incidence_angle(X, Y, Z, arr['NormalX'], arr['NormalY'], arr['NormalZ'])
    return R, np.arccos(theta)

def filter_by_incidence(arr, max_incidence):
    return apply_filter(arr, 'range', limits=f'Incidence[:{max_incidence}]')

def filter_normal(arr, knn):
    pipeline_json = {
        "pipeline": [
            {
                "type": "filters.normal",
                "knn": knn
            }
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json), arrays=[arr])
    pipeline.execute()
    return pipeline.arrays[0]

def process_lidar_data(las_file_name, traj_file_name, proj_code, reference_range):
    """Process LiDAR data to correct intensity and calculate incidence angle."""
    try:
        arr = read_las_file(las_file_name)
        trajectory_data = read_trajectory_data(traj_file_name)
        if arr is None or trajectory_data is None:
            return

        indices = list(normalize_intensity(arr['GpsTime'], trajectory_data[:,0]))
        arr['Range'], arr['Incidence'] = calculate_range_and_incidence(arr, trajectory_data, indices)

        # Filter incidence to less than 0.698132 (40 degrees) 
        arr = filter_by_incidence(arr, max_incidence=0.698132)
        
        # Correct intensity
        arr['CorrIntens'] = correct_intensity(arr['Intensity'], arr['Range'], reference_range, arr['Incidence'])

        # Write to LAS file
        corrected_filename = las_file_name[:-4] + '_corrected.las'
        write_las_file(arr, corrected_filename, proj_code)
        print('Completed for ' + las_file_name)
        
    except Exception as e:
        logging.error(f"Error processing LiDAR data for {las_file_name}: {e}")

        

def main():
    args = parse_arguments()
    
    # Configuration parameters from command-line arguments
    file_path = args.file_path
    input_laz_file = args.input_laz_file
    epsg_code = args.epsg_code
    trajectory_csv = args.trajectory_csv
    reference_range = args.reference_range
    min_scan_angle = args.min_scan_angle
    max_scan_angle = args.max_scan_angle
    
    # Log the configurations
    logging.info(f"Using file_path: {file_path}")
    logging.info(f"Using input_laz_file: {input_laz_file}")
    logging.info(f"Using EPSG code: {epsg_code}")
    logging.info(f"Using trajectory CSV: {trajectory_csv}")
    logging.info(f"Reference range: {reference_range}")
    logging.info(f"Scan angle limits: {min_scan_angle} to {max_scan_angle}")

    try:
        # Process each flight line
        gps_times_with_names = [
            {"name": "sbet_h1", "start": 236496.551092, "end": 236524.737629},
            {"name": "sbet_h2", "start": 236335.333763, "end": 236377.978642},
            {"name": "sbet_h3", "start": 236115.292836, "end": 236144.689638},
            {"name": "sbet_h4", "start": 235961.777337, "end": 236003.877066},
            {"name": "sbet_h5", "start": 235760.980922, "end": 235789.902605},
            {"name": "sbet_v6", "start": 233880.192683, "end": 233923.682649},
            {"name": "sbet_v5", "start": 234048.506281, "end": 234092.626409},
            {"name": "sbet_v4", "start": 234156.020964, "end": 234201.541418},
            {"name": "sbet_v3", "start": 234363.123551, "end": 234406.178455},
            {"name": "sbet_v2", "start": 234480.93065, "end": 234524.115591},
            {"name": "sbet_v1", "start": 234694.554843, "end": 234736.974615},
        ]
        process_all_point_clouds(gps_times_with_names, input_laz_file, epsg_code, file_path, min_scan_angle, max_scan_angle)

            
        # Process trajectories
        trajectory_times = process_trajectory(file_path, trajectory_csv)
        
        # Proceed only if there’s a match between the LAS files and trajectory files
        if len(trajectory_times) > 0:
            for las_f_name, traj_f_name in zip(glob.glob(os.path.join(file_path, '*.las')), trajectory_times):
                process_lidar_data(las_f_name, traj_f_name, epsg_code, epsg_code, reference_range)

    except Exception as e:
        logging.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()