import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from tqdm import tqdm
import matplotlib.ticker

from sg.grids import CubeSphere, StretchedGrid

from pcolormesh2 import pcolormesh2


def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), **kwargs, linestyle='-')


filename = '/home/liam/analysis/ensemble-2/c180e.nc'
layer = 35
IDs = ['NA1', 'EU1', 'IN1', 'SE1']
cmap = plt.get_cmap('Dark2')
colors = [cmap(0), cmap(1), cmap(2), cmap(3)]
species = 'N2O'

ds = xr.open_dataset(filename)
grid = CubeSphere(180)

fig = plt.figure(figsize=(8, 8))

gs = plt.GridSpec(3, 1, height_ratios=[10,10,1], left=0.05, right=0.95, top=0.98, bottom=0.07, wspace=0.01, hspace=0.1)

ax = fig.add_subplot(gs[0,0],  projection=ccrs.EqualEarth())
ax.coastlines()
ax.set_global()


da_ctl = ds[species].isel(lev=layer).sel(ID='CTL').squeeze()
cmap = 'cividis'
norm = plt.Normalize(float(da_ctl.quantile(0.05)), float(da_ctl.quantile(0.95)))
for face in tqdm(range(6)):
    pcolormesh2(grid.xe(face), grid.ye(face), da_ctl.isel(face=face), 180 if face != 2 else 20, norm, cmap=cmap)

da_sg = {}
sg = {}

for ID, c in zip(IDs, colors):
    ds_sg = ds.sel(ID=ID)
    da_sg[ID] = ds_sg[species].isel(lev=layer).squeeze()
    sg[ID] = StretchedGrid(int(ds_sg.cs_res), float(ds_sg.stretch_factor), float(ds_sg.target_lat),
                                   float(ds_sg.target_lon))
    draw_major_grid_boxes_naive(plt.gca(), sg[ID].xe(5), sg[ID].ye(5), color=c, linewidth=2.5)

ax = fig.add_subplot(gs[1,0],  projection=ccrs.EqualEarth())
ax.coastlines()
ax.set_global()

for ID, c in zip(IDs, colors):
    for face in tqdm(range(6)):
        pc = pcolormesh2(grid.xe(face), grid.ye(face), da_sg[ID].isel(face=face), 180 if face != 2 else 20, norm, cmap=cmap)
    draw_major_grid_boxes_naive(plt.gca(), sg[ID].xe(5), sg[ID].ye(5), color=c, linewidth=2.5, label=ID)

# plt.legend()
ax = fig.add_subplot(gs[2,0])
plt.colorbar(pc, orientation='horizontal', cax=ax)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.setp(ax.get_xticklabels(), visible=False)
# plt.setp(ax.get_yticklabels(), visible=False)
# ax.get_yaxis().set_visible(False)
# ax.get_xaxis().set_visible(False)
plt.tight_layout()
print('Saving to file...')
plt.savefig(f'{species}-{layer}.png', dpi=300)
print('Done!')
# plt.show()