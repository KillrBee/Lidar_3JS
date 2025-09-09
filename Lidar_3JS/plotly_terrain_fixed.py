#!/usr/bin/env python3
"""
Fixed Plotly-based Lidar Terrain Visualization - Addresses rendering issues
Web-based rendering with proper surface generation and auto-browser opening.
Enhanced with bug fixes and performance improvements.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.merge import merge
from rasterio import transform as rt
from skimage import measure
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import webbrowser
import os
import gc
from scipy.interpolate import griddata


@dataclass
class PlotlyTerrainConfig:
    """Configuration for Plotly terrain visualization."""
    
    # Input data paths - use Path objects for better path handling
    tile_index_path: Path = Path("/Users/gregorybroadhead/Lidar_Project/OntarioDTM_LidarDerived_TileIndex/OntarioDTM_LidarDerived_TileIndex.shp")
    tile_directory: Path = Path("/Users/gregorybroadhead/Lidar_Project/HuronGeorgianBay-DTM-29")
    
    # Target location
    target_longitude: float = -79.90776
    target_latitude: float = 44.75055
    
    # Processing parameters
    contour_interval: float = 1.0
    save_contour_interval: float = 0.5
    vertical_exaggeration: float = 2.0
    
    # Output settings
    cache_mosaic: bool = True
    mosaic_cache_path: Path = Path("./cached_mosaic.tif")
    
    # Visualization settings
    terrain_opacity: float = 0.9  # Increased opacity
    colorscale: str = "earth"  # Fixed: use valid Plotly colorscales
    height: int = 800
    width: int = 1200
    max_points: int = 150  # Limit points for better rendering
    
    # Performance settings
    contour_stride: int = 2  # Use every nth contour level
    max_contours: int = 50  # Maximum number of contour lines
    interpolate_small_gaps: bool = True  # Interpolate small NaN regions
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        self.tile_index_path = Path(self.tile_index_path)
        self.tile_directory = Path(self.tile_directory)
        self.mosaic_cache_path = Path(self.mosaic_cache_path)


class PlotlyTerrainProcessor:
    """Fixed Plotly-based terrain processor with enhanced error handling."""
    
    def __init__(self, config: PlotlyTerrainConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _validate_dem_data(self, dem: np.ndarray) -> None:
        """Validate DEM data before processing."""
        if dem.size == 0:
            raise ValueError("Empty DEM array")
        
        valid_ratio = np.sum(~np.isnan(dem)) / dem.size
        if valid_ratio < 0.1:
            raise ValueError(f"DEM has only {valid_ratio*100:.1f}% valid data")
        
        if dem.shape[0] * dem.shape[1] > 1e8:  # 100 million points
            self.logger.warning("Very large DEM detected, consider increasing sampling step")
    
    def find_intersecting_tiles(self) -> List[str]:
        """Find DTM tiles that intersect with the target point."""
        self.logger.info("Loading tile index...")
        
        if not self.config.tile_index_path.exists():
            raise FileNotFoundError(f"Tile index not found: {self.config.tile_index_path}")
        
        index = gpd.read_file(str(self.config.tile_index_path))
        
        # Filter for DTM tiles from Huron-Georgian Bay
        dtm = index[index["Package"].str.contains("DTM", case=False, na=False)]
        hgb = dtm[dtm["Project"].str.contains("Huron-Georgian Bay", case=False, na=False)]
        
        # Create point geometry and find intersections
        pt = gpd.GeoSeries([Point(self.config.target_longitude, self.config.target_latitude)], 
                         crs="EPSG:4326").to_crs(hgb.crs)
        
        hit = gpd.sjoin(hgb, gpd.GeoDataFrame(geometry=pt), predicate="intersects")
        
        tiles = hit["FileName"].tolist()
        self.logger.info(f"Found {len(tiles)} intersecting tiles")
        return tiles
    
    def create_mosaic(self, tiles: List[str]) -> Tuple[np.ndarray, rasterio.Affine]:
        """Create a merged mosaic from tile files."""
        # Check for cached mosaic
        if self.config.cache_mosaic and self.config.mosaic_cache_path.exists():
            self.logger.info("Loading cached mosaic...")
            with rasterio.open(str(self.config.mosaic_cache_path)) as src:
                return src.read(1), src.transform
        
        # Build file paths
        tif_paths = []
        for tile in tiles:
            tile_path = self.config.tile_directory / tile
            if tile_path.exists():
                tif_paths.append(tile_path)
            else:
                self.logger.warning(f"Tile file not found: {tile_path}")
        
        if not tif_paths:
            raise FileNotFoundError("No valid tile files found")
        
        # Open and merge rasters
        srcs = []
        try:
            for path in tif_paths:
                srcs.append(rasterio.open(str(path)))
            
            self.logger.info("Creating mosaic...")
            mosaic, out_trans = merge(srcs)
            dem = mosaic[0]
            
            # Validate the DEM data
            self._validate_dem_data(dem)
            
            # Cache if requested
            if self.config.cache_mosaic:
                self._cache_mosaic(dem, out_trans, srcs[0].meta)
            
            return dem, out_trans
        finally:
            for src in srcs:
                src.close()
    
    def _cache_mosaic(self, dem: np.ndarray, transform: rasterio.Affine, meta: dict) -> None:
        """Cache the mosaic to disk."""
        meta_out = meta.copy()
        meta_out.update({
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'transform': transform,
            'count': 1
        })
        
        with rasterio.open(str(self.config.mosaic_cache_path), 'w', **meta_out) as dst:
            dst.write(dem, 1)
        
        self.logger.info(f"Mosaic cached to {self.config.mosaic_cache_path}")
    
    def prepare_terrain_data(self, dem: np.ndarray, transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare terrain data with proper sampling and NaN handling."""
        self.logger.info("Preparing terrain data for visualization...")
        
        n_rows, n_cols = dem.shape
        self.logger.info(f"Original DEM size: {n_rows} x {n_cols}")
        
        # Calculate sampling step to limit points
        step = max(1, max(n_rows, n_cols) // self.config.max_points)
        self.logger.info(f"Using sampling step: {step}")
        
        # Sample the DEM
        dem_sampled = dem[::step, ::step]
        sampled_rows, sampled_cols = dem_sampled.shape
        self.logger.info(f"Sampled DEM size: {sampled_rows} x {sampled_cols}")
        
        # Create coordinate arrays for sampled data
        row_indices = np.arange(0, n_rows, step)[:sampled_rows]
        col_indices = np.arange(0, n_cols, step)[:sampled_cols]
        
        # Create meshgrid
        rows_mg, cols_mg = np.meshgrid(row_indices, col_indices, indexing='ij')
        
        # Transform to real-world coordinates
        xs, ys = rt.xy(transform, rows_mg, cols_mg)
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Apply vertical exaggeration
        zs = dem_sampled * self.config.vertical_exaggeration
        
        # Enhanced NaN handling
        valid_mask = ~np.isnan(zs)
        n_valid = np.sum(valid_mask)
        n_total = zs.size
        
        if n_valid == 0:
            raise ValueError("DEM contains no valid elevation data")
        
        self.logger.info(f"Valid elevation points: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        
        # Handle NaN values with interpolation for small gaps
        if np.any(~valid_mask):
            nan_ratio = np.sum(~valid_mask) / n_total
            
            if self.config.interpolate_small_gaps and nan_ratio < 0.1:
                # Interpolate small gaps
                self.logger.info("Interpolating small gaps in elevation data...")
                try:
                    # Get valid points and values
                    valid_points = np.column_stack((xs[valid_mask], ys[valid_mask]))
                    valid_values = zs[valid_mask]
                    
                    # Get points to interpolate
                    all_points = np.column_stack((xs.ravel(), ys.ravel()))
                    
                    # Interpolate
                    interpolated = griddata(valid_points, valid_values, all_points, method='linear')
                    
                    # Reshape and fill NaN values that couldn't be interpolated
                    zs_interp = interpolated.reshape(zs.shape)
                    remaining_nans = np.isnan(zs_interp)
                    if np.any(remaining_nans):
                        zs_interp[remaining_nans] = np.median(valid_values)
                    
                    zs = zs_interp
                except Exception as e:
                    self.logger.warning(f"Interpolation failed: {e}, using median fill")
                    zs[~valid_mask] = np.median(zs[valid_mask])
            else:
                # Use median for larger gaps
                zs[~valid_mask] = np.median(zs[valid_mask])
        
        return xs, ys, zs
    
    def generate_contour_lines(self, dem: np.ndarray, transform: rasterio.Affine) -> List[dict]:
        """Generate contour lines for Plotly visualization with optimization."""
        self.logger.info("Generating contour lines...")
        
        # Handle nodata values
        valid_data = dem[~np.isnan(dem)]
        if len(valid_data) == 0:
            self.logger.warning("No valid elevation data for contours")
            return []
            
        min_elev = np.nanmin(valid_data)
        max_elev = np.nanmax(valid_data)
        
        # Generate only the contour levels we'll use
        levels = np.arange(
            min_elev, 
            max_elev + self.config.contour_interval * self.config.contour_stride, 
            self.config.contour_interval * self.config.contour_stride
        )
        
        # Limit number of contour levels
        if len(levels) > self.config.max_contours:
            levels = np.linspace(min_elev, max_elev, self.config.max_contours)
        
        self.logger.info(f"Generating {len(levels)} contour levels")
        
        contour_traces = []
        for i, z in enumerate(levels):
            try:
                contour_coords = measure.find_contours(dem, z)
                for coords in contour_coords:
                    if len(coords) > 3:  # Only significant contours
                        xs, ys = rt.xy(transform, coords[:, 0], coords[:, 1])
                        
                        # Apply vertical exaggeration to Z coordinates
                        zs = [z * self.config.vertical_exaggeration] * len(xs)
                        
                        contour_traces.append({
                            'x': list(xs),
                            'y': list(ys), 
                            'z': zs,
                            'elevation': z
                        })
            except Exception as e:
                self.logger.debug(f"Skipping contour at {z}m: {e}")
                continue
        
        self.logger.info(f"Generated {len(contour_traces)} contour line segments")
        return contour_traces
    
    def create_plotly_visualization(self, dem: np.ndarray, transform: rasterio.Affine) -> go.Figure:
        """Create improved interactive Plotly 3D visualization."""
        self.logger.info("Creating Plotly 3D visualization...")
        
        try:
            # Prepare terrain data with better handling
            xs, ys, zs = self.prepare_terrain_data(dem, transform)
            
            # Create main 3D surface with better settings
            surface = go.Surface(
                x=xs,
                y=ys,
                z=zs,
                colorscale=self.config.colorscale,
                opacity=self.config.terrain_opacity,
                name='Terrain',
                colorbar=dict(title="Elevation (m)"),
                hovertemplate="<b>Elevation:</b> %{z:.1f}m<br>" +
                             "<b>Easting:</b> %{x:.0f}m<br>" +
                             "<b>Northing:</b> %{y:.0f}m<extra></extra>",
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    specular=0.1
                ),
                # Ensure proper surface rendering
                contours=dict(
                    z=dict(show=False),
                    x=dict(show=False),
                    y=dict(show=False)
                )
            )
            
            # Generate contour lines
            contour_lines = self.generate_contour_lines(dem, transform)
            
            # Create figure
            fig = go.Figure()
            
            # Add terrain surface
            fig.add_trace(surface)
            
            # Add contour lines (limited for performance)
            for i, contour in enumerate(contour_lines):
                if i >= self.config.max_contours:
                    break
                fig.add_trace(go.Scatter3d(
                    x=contour['x'],
                    y=contour['y'],
                    z=contour['z'],
                    mode='lines',
                    line=dict(color='black', width=3),
                    name=f"Contour {contour['elevation']:.1f}m",
                    showlegend=False,
                    hovertemplate=f"<b>Contour:</b> {contour['elevation']:.1f}m<extra></extra>"
                ))
            
            # Configure layout with better settings
            fig.update_layout(
                title=dict(
                    text=f"3D Terrain Visualization<br><sub>Lat: {self.config.target_latitude:.5f}, "
                         f"Lon: {self.config.target_longitude:.5f} | "
                         f"Vertical Exaggeration: {self.config.vertical_exaggeration}x</sub>",
                    x=0.5,
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title="Easting (m)",
                    yaxis_title="Northing (m)",
                    zaxis_title="Elevation (m)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.0),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.6),
                    # Better lighting
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)")
                ),
                width=self.config.width,
                height=self.config.height,
                margin=dict(r=20, b=20, l=20, t=80),
                # Better rendering settings
                showlegend=False
            )
            
            return fig
            
        finally:
            # Clean up large arrays
            if 'xs' in locals():
                del xs
            if 'ys' in locals():
                del ys
            if 'zs' in locals():
                del zs
            gc.collect()
    
    def create_multiple_exaggeration_views(self, dem: np.ndarray, transform: rasterio.Affine) -> go.Figure:
        """Create subplot with multiple exaggeration levels."""
        self.logger.info("Creating multiple exaggeration views...")
        
        exaggeration_levels = [1.0, 2.0, 3.0, 5.0]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=[f"{exag}x Vertical Exaggeration" for exag in exaggeration_levels],
            horizontal_spacing=0.05,
            vertical_spacing=0.05
        )
        
        try:
            # Prepare base data
            xs, ys, zs_base = self.prepare_terrain_data(dem, transform)
            # Remove exaggeration from base data
            zs_base = zs_base / self.config.vertical_exaggeration
            
            # Add surface for each exaggeration level
            for i, exag in enumerate(exaggeration_levels):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                zs = zs_base * exag
                
                fig.add_trace(
                    go.Surface(
                        x=xs, y=ys, z=zs,
                        colorscale=self.config.colorscale,
                        showscale=(i == 0),  # Only show colorbar for first plot
                        name=f'{exag}x',
                        opacity=self.config.terrain_opacity,
                        lighting=dict(ambient=0.4, diffuse=0.8)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Terrain at Multiple Vertical Exaggerations",
                height=900,
                width=1400,
                showlegend=False
            )
            
            return fig
            
        finally:
            # Clean up
            if 'xs' in locals():
                del xs
            if 'ys' in locals():
                del ys
            if 'zs_base' in locals():
                del zs_base
            gc.collect()
    
    def process_and_visualize(self) -> None:
        """Main processing pipeline with auto-browser opening."""
        try:
            # Find tiles and create mosaic
            tiles = self.find_intersecting_tiles()
            dem, transform = self.create_mosaic(tiles)
            
            # Create main 3D visualization
            fig = self.create_plotly_visualization(dem, transform)
            
            # Save and show main visualization
            main_file = "terrain_3d_interactive.html"
            self.logger.info(f"Saving main visualization to {main_file}")
            fig.write_html(main_file)
            
            # Auto-open in browser
            self.logger.info("Opening main visualization in browser...")
            try:
                webbrowser.open(f"file://{os.path.abspath(main_file)}")
            except Exception as e:
                self.logger.warning(f"Could not auto-open browser: {e}")
            
            # Create multiple exaggeration views
            multi_fig = self.create_multiple_exaggeration_views(dem, transform)
            
            # Save and show multiple views
            multi_file = "terrain_multiple_views.html"
            self.logger.info(f"Saving multiple views to {multi_file}")
            multi_fig.write_html(multi_file)
            
            # Auto-open second visualization
            self.logger.info("Opening multiple views in browser...")
            try:
                webbrowser.open(f"file://{os.path.abspath(multi_file)}")
            except Exception as e:
                self.logger.warning(f"Could not auto-open browser: {e}")
            
            self.logger.info("✅ Plotly visualizations created successfully!")
            self.logger.info("📁 Files saved:")
            self.logger.info(f"   - {main_file} (main 3D view)")
            self.logger.info(f"   - {multi_file} (4 exaggeration levels)")
            
            # Print data statistics
            valid_points = np.sum(~np.isnan(dem))
            total_points = dem.size
            self.logger.info(f"📊 Data statistics:")
            self.logger.info(f"   - Total elevation points: {total_points:,}")
            self.logger.info(f"   - Valid elevation points: {valid_points:,} ({100*valid_points/total_points:.1f}%)")
            self.logger.info(f"   - Elevation range: {np.nanmin(dem):.1f}m to {np.nanmax(dem):.1f}m")
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Plotly Terrain Visualization - macOS Friendly")
    parser.add_argument('--lat', '--latitude', type=float, required=True, help='Target latitude (WGS84)')
    parser.add_argument('--lon', '--longitude', type=float, required=True, help='Target longitude (WGS84)')
    parser.add_argument('--exaggeration', type=float, default=2.0, help='Vertical exaggeration (default: 2.0)')
    parser.add_argument('--colorscale', 
                       choices=['earth', 'viridis', 'plasma', 'terrain', 'turbo', 'hot', 'jet'], 
                       default='earth', 
                       help='Color scheme (default: earth)')
    parser.add_argument('--opacity', type=float, default=0.9, help='Terrain opacity (default: 0.9)')
    parser.add_argument('--max-points', type=int, default=150, help='Max points per axis (default: 150)')
    parser.add_argument('--contour-stride', type=int, default=2, help='Use every nth contour level (default: 2)')
    parser.add_argument('--max-contours', type=int, default=50, help='Maximum number of contour lines (default: 50)')
    parser.add_argument('--no-interpolation', action='store_true', help='Disable interpolation of small gaps')
    
    args = parser.parse_args()
    
    try:
        config = PlotlyTerrainConfig()
        config.target_latitude = args.lat
        config.target_longitude = args.lon
        config.vertical_exaggeration = args.exaggeration
        config.colorscale = args.colorscale
        config.terrain_opacity = args.opacity
        config.max_points = args.max_points
        config.contour_stride = args.contour_stride
        config.max_contours = args.max_contours
        config.interpolate_small_gaps = not args.no_interpolation
        
        print("🌐 Using Fixed Plotly for macOS-friendly 3D visualization")
        print(f"📍 Location: {args.lat:.5f}, {args.lon:.5f}")
        print(f"📏 Vertical exaggeration: {args.exaggeration}x")
        print(f"🎨 Color scheme: {args.colorscale}")
        print(f"🔍 Max points per axis: {args.max_points}")
        print(f"👁️ Terrain opacity: {args.opacity}")
        print(f"📊 Contour settings: stride={args.contour_stride}, max={args.max_contours}")
        
        processor = PlotlyTerrainProcessor(config)
        processor.process_and_visualize()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
