from glob import glob
import os
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyproj import Transformer


"""
This code make trends for CARRA and CMIP6 projections for a define area. The array obtained is used to force clim_load_climate, in completion with the carra MST, MAT and precipitations.
"""


########################### GENERAL ##############################
xmin, xmax, ymin, ymax= [543500, 566500, 8633000, 8655000] ## NW HEER LAND


ssp = "5-8.5"  # ssp 1-2.6, 2-4.5, 5-8.5


print("*"*20, "SCENARIO", ssp, "*"*40)
##############################FUNCTIONS#############################"
def load_cmip6(path, variable):
    ds = xr.open_dataset(path)

    da = ds[variable]

    da = da.rio.write_crs("EPSG:4326")

    return da


def load_carra(folder, variable, xmin, xmax, ymin, ymax):
    
    lon_min, lon_max, lat_min, lat_max = utm_to_wgs84_extent(
        xmin, xmax, ymin, ymax
    )
    
    files = sorted(glob(os.path.join(folder, "*.nc")))

    dates = []
    values = []

    for file in files:

        ds = xr.open_dataset(file)
    
        
        basename = os.path.basename(file)

        _, year, month, *_ = basename.replace(".nc","").split("_")

        date = pd.Timestamp(
            year=int(year),
            month=int(month),
            day=15
        )

        ds = ds.expand_dims(time=[date])
        da = crop_carra(ds[variable],lon_min,lon_max,lat_min,lat_max)
        value = da.mean(dim=("y", "x")).item()
    
        dates.append(date)
        values.append(value)

        ds.close()
        ts = xr.DataArray(
            values,
            coords={"time": dates},
            dims="time",
            name=variable
        )
    


    return ts #ds[variable]


def reproject_crop(
        da,
        xmin,
        xmax,
        ymin,
        ymax,
        dst_crs="EPSG:32633",
        resampling=Resampling.bilinear
):

    da = da.rio.reproject(
        dst_crs,
        resampling=resampling
    )

    da = da.rio.clip_box(
        minx=xmin,
        miny=ymin,
        maxx=xmax,
        maxy=ymax
    )

    return da

def utm_to_wgs84_extent(xmin, xmax, ymin, ymax):
    """
    Convertit un extent UTM 33N en WGS84.
    Retourne :
        lon_min, lon_max, lat_min, lat_max
    """

    transformer = Transformer.from_crs(
        "EPSG:32633",
        "EPSG:4326",
        always_xy=True
    )

    # Coins de la boîte
    lon1, lat1 = transformer.transform(xmin, ymin)
    lon2, lat2 = transformer.transform(xmax, ymax)

    lon_min = min(lon1, lon2)
    lon_max = max(lon1, lon2)

    lat_min = min(lat1, lat2)
    lat_max = max(lat1, lat2)

    return lon_min, lon_max, lat_min, lat_max


def crop_carra(da, lon_min, lon_max, lat_min, lat_max):

    mask = (
        (da.latitude >= lat_min) &
        (da.latitude <= lat_max) &
        (da.longitude >= lon_min) &
        (da.longitude <= lon_max)
    )

    return da.where(mask)


def linear_trend(ts):

    years = ts.year.values

    values = ts.values

    result = linregress(years, values)

    return {

        "slope":result.slope,

        "intercept":result.intercept,

        "r2":result.rvalue**2,

        "p":result.pvalue,

        "trend":
            result.intercept+
            result.slope*years

    }


def plot_trend(var, years, trend):
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(years, var, "o-", label="Annual mean")
    ax.plot(
        years,
        trend,
        "r--",
        linewidth=2,
        label=f"Tendancy = {slope:.4f} °C/a"
    )

    ax.set_title("Mean annuaal temperature evolution and trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.text(
        0.02, 0.98,
        f"$R^2$ = {r2:.3f}\np = {pvalue:.3e}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )   

    plt.show()




########################## Code##############



if ssp == "1-2.6":
    path_cmip_T = "~/PhD_Lucie/DATA/CLIMATE/CMIP6/TEMP_ssp1-2.6/t2m_ssp126.nc"
elif ssp == "2-4.5":
    path_cmip_T = "~/PhD_Lucie/DATA/CLIMATE/CMIP6/TEMP_ssp2-4.5/t2m_ssp245.nc"
elif ssp == "5-8.5":
    path_cmip_T = "~/PhD_Lucie/DATA/CLIMATE/CMIP6/TEMP_ssp5-8.5/t2m_ssp585.nc"


da_T_cmip = load_cmip6(path_cmip_T,"tas")

da_T_cmip = reproject_crop(
    da_T_cmip,
    xmin,
    xmax,
    ymin,
    ymax
)

ts_cmip_monthly = da_T_cmip.mean(dim=("y", "x"))
ts_cmip_monthly = ts_cmip_monthly -273.15 ## in degree !

ts_cmip_annual = ts_cmip_monthly.groupby("time.year").mean()


#################### Lienar regression


years = ts_cmip_annual["year"].values
temps = ts_cmip_annual.values
result = linregress(years, temps)

slope = result.slope          # °C/an (ou K/an)
intercept = result.intercept
r2 = result.rvalue**2
pvalue = result.pvalue

# Valeurs de la droite
trend = intercept + slope * years

#plot_trend(temps, years, trend)

trend_T_2020_2100= slope*80
print("*"*30,f"trend Temp 2020 2100 = {trend_T_2020_2100:.3f}", "*"*30)





######################### Precips cmip6 ##############
if ssp == "1-2.6":
    path_cmip_P = "~/PhD_Lucie/DATA/CLIMATE/CMIP6/PRECIP_ssp1-2.6/precips_ssp126.nc"
elif ssp == "2-4.5":
    path_cmip_P = "~/PhD_Lucie/DATA/CLIMATE/CMIP6/PRECIP_ssp2-4.5/precips_ssp245.nc"
elif ssp == "5-8.5":
    path_cmip_P= "~/PhD_Lucie/DATA/CLIMATE/CMIP6/PRECIP_ssp5-8.5/precips_ssp5-85.nc"


da_P_cmip = load_cmip6(path_cmip_P,"pr")

da_P_cmip = reproject_crop(
    da_P_cmip,
    xmin,
    xmax,
    ymin,
    ymax
)

pr_cmip_monthly = da_P_cmip.mean(dim=("y", "x"))
pr_cmip_monthly = pr_cmip_monthly  
pr_cmip_annual = pr_cmip_monthly.groupby("time.year").sum()

years = pr_cmip_annual["year"].values
precs = pr_cmip_annual.values

# Régression linéaire
result = linregress(years, precs)

slope = result.slope          # °C/an (ou K/an)
intercept = result.intercept
r2 = result.rvalue**2
pvalue = result.pvalue

# Valeurs de la droite
trend = intercept + slope * years

#plot_trend(precs, years, trend)
trend_PR_2020_2100= slope*80

##### percentage evolution (tancr)
P_ref = pr_cmip_annual.sel(year=slice(2015, 2034)).mean()
P_fin = pr_cmip_annual.sel(year=slice(2079, 2100)).mean()
P_offset = 100 * P_fin / P_ref
trend_precip_2020_2100 = P_offset.values
print("*"*20,f"precipitation offset", trend_precip_2020_2100,"*"*20)

















############################################## CARRA ###############################

folder_T = "../CODES_TANCREDE_CLIMATE/carra_monthly_t2m/"



ts_cr_monthly = load_carra(
    folder_T,
    "t2m",
    xmin, xmax, ymin, ymax)


ts_cr_monthly = ts_cr_monthly -273.15 ## in degree !

ts_cr_annual = ts_cr_monthly.groupby("time.year").mean()

#################### Lienar regression


years = ts_cr_annual["year"].values
temps = ts_cr_annual.values
result = linregress(years, temps)

slope = result.slope          # °C/an (ou K/an)
intercept = result.intercept
r2 = result.rvalue**2
pvalue = result.pvalue

# Valeurs de la droite
trend = intercept + slope * years

plot_trend(temps, years, trend)

trend_T_1990_2020= slope*30
print("*"*30,f"trend Temp 1990 2020 = {trend_T_1990_2020:.3f}", "*"*30)

############################# PRECIPS CARRA
folder_P = "../CODES_TANCREDE_CLIMATE/carra_monthly_precipitation/"

pr_cr_monthly = load_carra(
    folder_P,
    "tp",xmin, xmax, ymin, ymax)




pr_cr_annual = pr_cr_monthly.groupby("time.year").sum()

#################### Lienar regression


years = pr_cr_annual["year"].values
precs = pr_cr_annual.values
result = linregress(years, precs)

slope = result.slope          # °C/an (ou K/an)
intercept = result.intercept
r2 = result.rvalue**2
pvalue = result.pvalue

# Valeurs de la droite
trend = intercept + slope * years

plot_trend(precs, years, trend)

trend_P_1990_2020= slope*30

##### percentage evolution (tancr)
P_ref = pr_cr_annual.sel(year=slice(1990, 2000)).mean()
P_fin = pr_cr_annual.sel(year=slice(2010, 2020)).mean()
P_offset = 100 * P_fin / P_ref
trend_precip_1990_2020 = P_offset.values
print("*"*20,f"precipitation offset", trend_precip_1990_2020,"*"*20)

########################################### BUILD array final ###########
print("\n")
print("="*20,"FOR IGM","="*20)
print("\n")

mid_precip = (trend_precip_1990_2020-100)/2

print(mid_precip)
print(f"- [1990, {-trend_T_1990_2020/2:.1f},", int(-mid_precip +100)  ,"]")
print(f"- [2020, {trend_T_1990_2020/2:.1f},", int(mid_precip +100)  ,"]")
print(f"- [2100, {trend_T_2020_2100 + trend_T_1990_2020/2:.1f},", int(trend_precip_2020_2100 +mid_precip) ,"]")

