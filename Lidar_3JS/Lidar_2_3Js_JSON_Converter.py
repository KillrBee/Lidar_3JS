#!/usr/bin/env python3
"""
LIDAR Terrain Data Processor for three.js
This script processes LIDAR DTM tiles, creates a mosaic, calculates contour lines,
and exports the data to a JSON file suitable for a three.js web application.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.merge import merge
from rasterio import transform as rt
from skimage import measure
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import gc

# --- Configuration ---
class TerrainConfig:
    """Configuration for LIDAR data processing."""
    # --- Paths to your local data ---
    TILE_INDEX_PATH: Path = Path("../Terrain_Data/OntarioDTM_LidarDerived_TileIndex/OntarioDTM_LidarDerived_TileIndex.shp")
    TILE_DIRECTORY: Path = Path("../Terrain_Data/HuronGeorgianBay-DTM-29")

    # --- Target Location ---
    TARGET_LONGITUDE: float = -79.90776
    TARGET_LATITUDE: float = 44.75055

    # --- Processing Parameters ---
    # Increased max points for very high resolution.
    MAX_POINTS: int = 4096 * 4096
    # The interval in meters for the generated contour lines.
    CONTOUR_INTERVAL: float = 1.0

    # --- Output Settings ---
    OUTPUT_JSON_PATH: Path = Path("./centered_terrain_data.json")

# --- Main Processor Class ---
class LidarProcessor:
    """Processes LIDAR data and exports it for web visualization."""

    def __init__(self, config: TerrainConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def find_intersecting_tiles(self) -> List[str]:
        self.logger.info("Loading tile index...")
        if not self.config.TILE_INDEX_PATH.exists():
            raise FileNotFoundError(f"Tile index shapefile not found at: {self.config.TILE_INDEX_PATH}")
        index_gdf = gpd.read_file(self.config.TILE_INDEX_PATH)
        target_point = Point(self.config.TARGET_LONGITUDE, self.config.TARGET_LATITUDE)
        point_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs="EPSG:4326").to_crs(index_gdf.crs)
        intersecting_tiles_gdf = gpd.sjoin(index_gdf, point_gdf, predicate="intersects")
        if intersecting_tiles_gdf.empty:
            raise ValueError("No intersecting tiles found for the given coordinates.")
        tiles = intersecting_tiles_gdf["FileName"].tolist()
        self.logger.info(f"Found {len(tiles)} intersecting tiles: {tiles}")
        return tiles

    def create_mosaic(self, tiles: List[str]) -> Tuple[np.ndarray, rasterio.Affine]:
        self.logger.info("Creating mosaic from intersecting tiles...")
        sources_to_merge = []
        for tile_name in tiles:
            tile_path = self.config.TILE_DIRECTORY / tile_name
            if tile_path.exists():
                sources_to_merge.append(rasterio.open(tile_path))
            else:
                self.logger.warning(f"Tile file not found and will be skipped: {tile_path}")
        if not sources_to_merge:
            raise FileNotFoundError("No valid tile files were found to create a mosaic.")
        mosaic, out_trans = merge(sources_to_merge)
        for src in sources_to_merge:
            src.close()
        return mosaic[0], out_trans

    def prepare_terrain_data(self, dem: np.ndarray, transform: rasterio.Affine) -> Dict[str, Any]:
        self.logger.info("Preparing terrain data for three.js...")
        height, width = dem.shape
        self.logger.info(f"Original DEM size: {width}x{height}")

        step = max(1, int(np.sqrt((height * width) / self.config.MAX_POINTS)))
        self.logger.info(f"Using downsampling step: {step} to aim for ~{self.config.MAX_POINTS} points.")

        dem_sampled = dem[::step, ::step]
        sampled_height, sampled_width = dem_sampled.shape
        self.logger.info(f"Sampled DEM size: {sampled_width}x{sampled_height}")
        
        # Replace NaN values before processing
        dem_sampled[np.isnan(dem_sampled)] = 0
        min_z, max_z = dem_sampled.min(), dem_sampled.max()

        # --- Generate Vertices ---
        rows = np.arange(sampled_height) * step
        cols = np.arange(sampled_width) * step
        cols_mg, rows_mg = np.meshgrid(cols, rows)
        xs, ys = rt.xy(transform, rows_mg, cols_mg, offset='center')
        zs = dem_sampled

        # Calculate center point and store it for GPS conversion
        x_center, y_center = np.mean(xs), np.mean(ys)
        xs_centered, ys_centered = np.array(xs) - x_center, np.array(ys) - y_center
        
        # Store the transform and center for coordinate conversion
        terrain_bounds = {
            "minX": float(np.min(xs)),
            "maxX": float(np.max(xs)),
            "minY": float(np.min(ys)), 
            "maxY": float(np.max(ys)),
            "centerX": float(x_center),
            "centerY": float(y_center)
        }
        
        vertices = np.vstack((xs_centered.flatten(), ys_centered.flatten(), zs.flatten())).T.flatten().tolist()
        
        # --- Generate Contour Lines ---
        self.logger.info("Generating contour lines...")
        contours_data = []
        start_level = np.floor(min_z / self.config.CONTOUR_INTERVAL) * self.config.CONTOUR_INTERVAL
        
        for level in np.arange(start_level, max_z, self.config.CONTOUR_INTERVAL):
            contours = measure.find_contours(dem_sampled, level)
            for contour in contours:
                # Convert pixel coords (row, col) to real-world centered coords
                contour_rows = contour[:, 0] * step
                contour_cols = contour[:, 1] * step
                contour_xs, contour_ys = rt.xy(transform, contour_rows, contour_cols, offset='center')
                contour_xs_centered = np.array(contour_xs) - x_center
                contour_ys_centered = np.array(contour_ys) - y_center
                
                # Flatten to [x1, y1, z1, x2, y2, z2, ...]
                contour_vertices = np.vstack(
                    (contour_xs_centered, contour_ys_centered, np.full_like(contour_xs_centered, level))
                ).T.flatten().tolist()

                contours_data.append({
                    "level": level,
                    "vertices": contour_vertices
                })
        self.logger.info(f"Generated {len(contours_data)} contour lines.")

        del dem, dem_sampled, xs, ys, zs, xs_centered, ys_centered
        gc.collect()

        self.logger.info(f"Generated {len(vertices) // 3} vertices.")

        return {
            "width": sampled_width,
            "height": sampled_height,
            "vertices": vertices,
            "minElevation": float(min_z),
            "maxElevation": float(max_z),
            "contours": contours_data,
            "terrainBounds": terrain_bounds,  # NEW: Include terrain bounds and center
            "coordinateSystem": "centered_utm_zone_17n"  # NEW: Document coordinate system
        }

    def save_to_json(self, data: Dict[str, Any]):
        self.logger.info(f"Saving terrain data to {self.config.OUTPUT_JSON_PATH}...")
        with open(self.config.OUTPUT_JSON_PATH, 'w') as f:
            json.dump(data, f)
        self.logger.info("Save complete.")

    def run(self):
        try:
            tiles = self.find_intersecting_tiles()
            dem, transform = self.create_mosaic(tiles)
            terrain_data = self.prepare_terrain_data(dem, transform)
            self.save_to_json(terrain_data)
            self.logger.info("Processing finished successfully!")
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"An error occurred: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    config = TerrainConfig()
    processor = LidarProcessor(config)
    processor.run()