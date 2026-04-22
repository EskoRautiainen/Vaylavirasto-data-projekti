import time
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import LineString, MultiLineString


# ----------------------------
# PATH SETUP
# ----------------------------
try:
    codes_dir = Path(__file__).resolve().parent
except NameError:
    codes_dir = Path.cwd()

project_root = codes_dir.parent
output_folder = project_root / "output"
output_folder.mkdir(exist_ok=True)

gpkg_path = output_folder / "road_condition.gpkg"
csv_log_path = output_folder / "performance_results.csv"


# ----------------------------
# HELPERS
# ----------------------------
def count_vertices(geom):
    """Count total coordinate vertices in a geometry."""
    if geom is None or geom.is_empty:
        return 0

    if isinstance(geom, LineString):
        return len(geom.coords)

    if isinstance(geom, MultiLineString):
        return sum(len(part.coords) for part in geom.geoms)

    return 0


def prepare_gdf_for_web(gdf, keep_cols=None, simplify_tolerance=None):
    """
    Prepare a web-friendly GeoDataFrame:
    - keep only selected columns
    - convert CRS if needed
    - optionally simplify geometry
    """
    gdf = gdf[gdf.geometry.notnull()].copy()

    if keep_cols is not None:
        existing_cols = [col for col in keep_cols if col in gdf.columns]
        if "geometry" not in existing_cols:
            existing_cols.append("geometry")
        gdf = gdf[existing_cols].copy()

    # Ensure WGS84 for web usage
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set.")
    gdf = gdf.to_crs(epsg=4326)

    # Simplify in metric CRS
    simplify_time_s = 0.0
    if simplify_tolerance is not None:
        t0 = time.perf_counter()

        gdf = gdf.to_crs(epsg=3857).copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
        gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

        simplify_time_s = time.perf_counter() - t0

    return gdf, simplify_time_s


def build_map(gdf, html_path, line_weight=1.0, tooltip_fields=None):
    """
    Build and save a folium map from GeoDataFrame.
    Returns save time and output HTML size in MB.
    """
    finland_bounds = [
        [59.5, 19.0],
        [70.5, 32.5]
    ]

    m = folium.Map(
        location=[64.5, 26.0],
        zoom_start=5,
        tiles="OpenStreetMap",
        max_bounds=True
    )

    m.fit_bounds(finland_bounds)

    def style_function(feature):
        color = feature["properties"].get("color_hex", "#808080")
        return {
            "color": color,
            "weight": line_weight,
            "opacity": 0.75
        }

    tooltip = None
    if tooltip_fields:
        existing_tooltip_fields = [col for col in tooltip_fields if col in gdf.columns]
        if existing_tooltip_fields:
            tooltip = folium.GeoJsonTooltip(
                fields=existing_tooltip_fields,
                aliases=existing_tooltip_fields,
                localize=True,
                sticky=False
            )

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=tooltip,
        name="Road condition"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    t0 = time.perf_counter()
    m.save(html_path)
    save_time_s = time.perf_counter() - t0

    html_size_mb = html_path.stat().st_size / (1024 * 1024)

    return save_time_s, html_size_mb


def benchmark_case(
    base_gdf,
    case_name,
    simplify_tolerance=None,
    line_weight=1.0,
    tooltip_fields=None,
    keep_cols=None
):
    """
    Run one benchmark case and return results as a dict.
    """
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_name": case_name,
        "simplify_tolerance": simplify_tolerance,
        "line_weight": line_weight,
    }

    # Feature and vertex counts before optimization
    result["feature_count_before"] = len(base_gdf)
    result["vertices_before"] = int(base_gdf.geometry.apply(count_vertices).sum())

    # Prepare optimized GDF
    gdf_web, simplify_time_s = prepare_gdf_for_web(
        base_gdf,
        keep_cols=keep_cols,
        simplify_tolerance=simplify_tolerance
    )
    result["simplify_time_s"] = round(simplify_time_s, 4)

    # Feature and vertex counts after optimization
    vertex_counts_after = gdf_web.geometry.apply(count_vertices)
    result["feature_count_after"] = len(gdf_web)
    result["vertices_after"] = int(vertex_counts_after.sum())
    result["avg_vertices_after"] = round(float(vertex_counts_after.mean()), 2) if len(vertex_counts_after) else 0.0

    # Build HTML
    html_path = output_folder / f"benchmark_{case_name}.html"
    save_time_s, html_size_mb = build_map(
        gdf_web,
        html_path=html_path,
        line_weight=line_weight,
        tooltip_fields=tooltip_fields
    )

    result["html_save_time_s"] = round(save_time_s, 4)
    result["html_size_mb"] = round(html_size_mb, 2)
    result["html_file"] = html_path.name

    return result


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not gpkg_path.exists():
        print(f"GeoPackage not found: {gpkg_path}")
        return

    print(f"Reading GeoPackage: {gpkg_path}")

    t0 = time.perf_counter()
    gdf = gpd.read_file(gpkg_path)
    read_time_s = time.perf_counter() - t0

    print(f"Read time: {read_time_s:.3f} s")
    print(f"Rows loaded: {len(gdf)}")

    keep_cols = [
        "Tie",
        "Aosa",
        "AJORATA",
        "Kaista",
        "lane_index",
        "AET",
        "LET",
        "RMS_yhd",
        "color_hex",
        "geometry"
    ]

    tooltip_fields = ["Tie", "Aosa", "Kaista", "RMS_yhd"]

    benchmark_results = []

    # Baseline
    benchmark_results.append(benchmark_case(
        base_gdf=gdf,
        case_name="baseline",
        simplify_tolerance=None,
        line_weight=2.0,
        tooltip_fields=["Tie", "Aosa", "AJORATA", "lane_index", "AET", "LET", "RMS_yhd", "color_hex"],
        keep_cols=keep_cols
    ))

    # Lighter styling, smaller tooltip
    benchmark_results.append(benchmark_case(
        base_gdf=gdf,
        case_name="light_style",
        simplify_tolerance=None,
        line_weight=1.0,
        tooltip_fields=tooltip_fields,
        keep_cols=keep_cols
    ))

    # Simplification tests
    for tol in [1, 3, 5, 10]:
        benchmark_results.append(benchmark_case(
            base_gdf=gdf,
            case_name=f"simplify_{tol}",
            simplify_tolerance=tol,
            line_weight=1.0,
            tooltip_fields=tooltip_fields,
            keep_cols=keep_cols
        ))

    # Convert to DataFrame
    df_results = pd.DataFrame(benchmark_results)

    # Add common read time to all rows
    df_results["gpkg_read_time_s"] = round(read_time_s, 4)

    # Append to CSV if it exists, otherwise create it
    if csv_log_path.exists():
        existing = pd.read_csv(csv_log_path)
        df_results = pd.concat([existing, df_results], ignore_index=True)

    df_results.to_csv(csv_log_path, index=False)

    print("\nBenchmark results:")
    print(df_results.tail(len(benchmark_results)))
    print(f"\nSaved benchmark log to: {csv_log_path}")


if __name__ == "__main__":
    main()