import geopandas as gpd
from shapely.geometry import Polygon

selected_year = 1936
target_crs = "EPSG:32633"

# =========================================================
# 1. IMPORT REGIONAL AND GLACIER MASK
# =========================================================



region_path = "~/PhD_Lucie/DATA/Maps/Shapefiles_regions/mask_heer_land.shp"
gdf_region = gpd.read_file(region_path).to_crs(target_crs)
region_geom = gdf_region.unary_union

# ---------------------------------------------------------
# IMPORT GLACIER OUTLINES
# ---------------------------------------------------------


if selected_year == 1936:
    glacier_path = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1936-1972/CryoClim_GAO_SJ_1936-1972.shp"
    gdf_glaciers = gpd.read_file(glacier_path).to_crs(target_crs)
elif selected_year == 1990:
    glacier_path = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1990/CryoClim_GAO_SJ_1990.shp"
    gdf_glaciers = gpd.read_file(glacier_path).to_crs(target_crs)

gdf_glaciers = gdf_glaciers[gdf_glaciers.intersects(region_geom)] ## Only heerland


# =========================================================
# 2. IMPORT TXT FILE WITH MARINE-TERMINATING GLACIERS
# =========================================================

txt_path = "./list_marine_terminating_heer_land.txt"

with open(txt_path, "r") as f:
    glaciers = [line.strip() for line in f.readlines()]


print("LIST", glaciers)


gdf_marine = gdf_glaciers[gdf_glaciers["NAME"].isin(glaciers)]
gdf_land = gdf_glaciers[~gdf_glaciers["NAME"].isin(glaciers)] # ~ for contrary

print("GDF MARINE",   gdf_marine)
#print("GDF LAND",   gdf_land)


# =========================================================
# 3. SAVE FILES
# =========================================================


marine_out = f"marine_terminating_{selected_year}.shp"
land_out = f"land_terminating_{selected_year}.shp"

gdf_marine.to_file(marine_out)
gdf_land.to_file(land_out)

print("Marine mask saved :", marine_out)
print("Land mask saved :", land_out)
