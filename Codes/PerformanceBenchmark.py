import time
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import LineString, MultiLineString


# ----------------------------
# SETTINGS
# ----------------------------

# This note is set inside RunAll.py and can be used to add context to the performance log
RUN_NOTE = os.environ.get("RUN_NOTE", "no note")
OPEN_BROWSER = True  # Set to True to automatically open the performance log in a browser after running

#---------------------------
# PATHS
#---------------------------

try:
    codes_dir = Path(__file__).resolve().parent
except NameError:
    codes_dir = Path.cwd()

project_root = codes_dir.parent
output_folder = project_root / "output"
output_folder.mkdir(exist_ok=True)

gpkg_path = output_folder / "road_condition.gpkg"
html_path = output_folder / "road_condition_map.html"
csv_log_path = output_folder / "performance_results.csv"


# ----------------------------
# HELPERS
# ----------------------------
def count_vertices(geom):
    if geom is None or geom.is_empty:
        return 0

    if isinstance(geom, LineString):
        return len(geom.coords)

    if isinstance(geom, MultiLineString):
        return sum(len(part.coords) for part in geom.geoms)

    return 0


def file_size_mb(path):
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not gpkg_path.exists():
        print(f"Missing GeoPackage: {gpkg_path}")
        return

    if not html_path.exists():
        print(f"Missing HTML map: {html_path}")
        return

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": RUN_NOTE,
        "gpkg_file": gpkg_path.name,
        "html_file": html_path.name,
        "gpkg_size_mb": file_size_mb(gpkg_path),
        "html_size_mb": file_size_mb(html_path),
    }

    # Time GeoPackage read
    t0 = time.perf_counter()
    gdf = gpd.read_file(gpkg_path)
    result["gpkg_read_time_s"] = round(time.perf_counter() - t0, 4)

    # Basic geometry cleanup
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    # Feature counts
    result["feature_count"] = len(gdf)

    # Vertex counts
    vertex_counts = gdf.geometry.apply(count_vertices)
    result["total_vertices"] = int(vertex_counts.sum())
    result["avg_vertices_per_feature"] = round(float(vertex_counts.mean()), 2) if len(vertex_counts) else 0.0
    result["max_vertices_in_feature"] = int(vertex_counts.max()) if len(vertex_counts) else 0

    # Column count
    result["column_count"] = len(gdf.columns)

    # Optional useful counts if columns exist
    if "Tie" in gdf.columns:
        result["unique_tie_count"] = int(gdf["Tie"].nunique())

    if "Kaista" in gdf.columns:
        result["unique_kaista_count"] = int(gdf["Kaista"].nunique())

    if "AJORATA" in gdf.columns:
        result["unique_ajorata_count"] = int(gdf["AJORATA"].nunique())

    # Append to CSV
    df_new = pd.DataFrame([result])

    if csv_log_path.exists():
        df_old = pd.read_csv(csv_log_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_log_path, index=False)

    print("Performance record added:")
    print(df_new.to_string(index=False))
    print(f"\nSaved log to: {csv_log_path}")

    if OPEN_BROWSER:
        open_csv_as_html(df_all, csv_log_path)

def open_csv_as_html(df, output_path):
    html_path = output_path.with_suffix(".html")

    # Create a styled HTML table
    html = df.style.highlight_min(
        subset=["html_size_mb", "total_vertices"],
        color="#d4edda"
    ).to_html()

    # Optional: add simple styling
    styled_html = f"""
    <html>
    <head>
        <title>Performance Results</title>
        <style>
            body {{
                font-family: "Segoe UI", Arial, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
            }}

            h2 {{
                margin-bottom: 10px;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
                background-color: white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}

            th, td {{
                padding: 4px 8px;   /* 👈 smaller = more compact */
                text-align: left;
                border-bottom: 1px solid #eee;
                white-space: nowrap;
            }}

            th {{
                background-color: #343a40;
                color: white;
                position: sticky;
                top: 0;
                z-index: 1;
            }}

            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}

            tr:hover {{
                background-color: #e9f5ff;
            }}

            /* Make table scrollable instead of stretching vertically */
            .table-container {{
                max-height: 80vh;
                overflow-y: auto;
                border: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <h2>Performance Results</h2>
        <div class="table-container">
            {html}
        </div>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)

    print(f"Opened performance table: {html_path}")

    # Open automatically in browser
    import webbrowser
    webbrowser.open(html_path)

if __name__ == "__main__":
    main()
