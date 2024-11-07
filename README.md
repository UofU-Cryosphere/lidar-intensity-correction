# Lidar Intensity Correction
Tthe initial script (lidar_intenstiy_correction.py) calculates range and incidence angle before correcting the lidar intensity data per flight line. The corrected intensity is then converted to reflectance and used to calculate grain size (convert_to_grain_size.py).

## Requirements

- Python 3.x
- PDAL, NumPy, LasPy, Pandas, GeoPandas, and other dependencies listed in `environment.yml`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/lidar-processing
   cd lidar-processing

2. Set up the environment: Create a conda environment using environment.yml:
conda env create -f environment.yml

3. Activate the environment:
conda activate lidar-env

## Usage
Run the script with the following command-line arguments:
python main.py --file_path <path_to_data_directory> --input_laz_file <input_laz_file> \
               --epsg_code <EPSG_code> --trajectory_csv <trajectory_csv_file> \
               --reference_range <reference_range> --min_scan_angle <min_angle> \
               --max_scan_angle <max_angle>

## Example Usage
python main.py --file_path "./data" --input_laz_file "input.laz" \
               --epsg_code "EPSG:32613" --trajectory_csv "trajectory.csv" \
               --reference_range 1339 --min_scan_angle -15 --max_scan_angle 15
