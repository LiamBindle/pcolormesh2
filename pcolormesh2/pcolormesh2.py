import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry


def get_am_and_pm_masks_and_polygons_outline(xe, ye, far_from_pm=80):
    if np.any(xe >= 180):
        raise ValueError('xe must be in [-180, 180)')
    # xe must be [-180 to 180]
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xe[p0, p0]), np.sign(xe[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xe[p1, p0]), np.sign(xe[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xe[p1, p1]), np.sign(xe[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xe[p0, p1]), np.sign(xe[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    # Make xy polygons for each gridbox
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1]]), 0, -1)
    polygon_outlines = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)
    am = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)

    # Figure out which polygon_outlines cross the prime meridian and antimeridian
    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(polygon_outlines[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(far_from_pm, -90), (80, far_from_pm)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False

    return am, pm, polygon_outlines


def _pcolormesh2_internal(ax, X, Y, C, cmap, norm):
    X[X >= 180] -= 360

    am, pm, boxes_xy_pc = get_am_and_pm_masks_and_polygons_outline(X, Y)

    center_i = int(X.shape[0] / 2)
    center_j = int(X.shape[1] / 2)
    cX = X[center_i, center_j]
    cY = Y[center_i, center_j]

    gnomonic_crs = ccrs.Gnomonic(cY, cX)
    gnomonic_proj = pyproj.Proj(gnomonic_crs.proj4_init)

    X_gno, Y_gno = gnomonic_proj(X, Y)
    boxes_xy_gno = np.moveaxis(gnomonic_proj(boxes_xy_pc[..., 0], boxes_xy_pc[..., 1]), 0, -1)

    if np.any(np.isnan(X_gno)) or np.any(np.isnan(Y_gno)):
        raise ValueError('Block size is too big!')
    else:
        plt.pcolormesh(X_gno, Y_gno, np.ma.masked_array(C, ~am), transform=gnomonic_crs, cmap=cmap, norm=norm)

        for idx in np.argwhere(~am):
            c = cmap(norm(C[idx[0], idx[1]]))
            ax.add_geometries(
                [shapely.geometry.Polygon(boxes_xy_gno[idx[0], idx[1],...])],
                gnomonic_crs, edgecolor=c, facecolor=c
            )


def pcolormesh2(X, Y, C, blocksize, norm, **kwargs):
    kwargs.setdefault('cmap', 'viridis')
    cmap = plt.get_cmap(kwargs['cmap'])

    ax = plt.gca()

    for si, ei in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[0] // blocksize)]:
        for sj, ej in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[1] // blocksize)]:
            _pcolormesh2_internal(ax,
                X[si:ei, sj:ej],
                Y[si:ei, sj:ej],
                C[si:ei - 1, sj:ej - 1],
                cmap, norm
            )


if __name__ == '__main__':
    import xarray as xr

    plt.figure(figsize=(8,4))

    # ds = xr.open_dataset('/extra-space/diag_c90.nc')
    # grid = StretchedGrid(90, 1.0, -90, 170)
    # ds = xr.open_dataset('/extra-space/regridded_s24.nc')
    # grid = StretchedGrid(24, 4.0, 40, 250)
    # ds = xr.open_dataset('/extra-space/regridded_c24.nc')
    # grid = StretchedGrid(24, 1.0, -90, 170)
    ds = xr.open_dataset('/extra-space/sg-stats/Sept-2/S48/GCHP.SpeciesConc.Sept.nc', decode_times=False)
    grid = xr.open_dataset('/extra-space/sg-stats/Sept/S48/grid_box_outlines_and_centers.nc')

    da = (ds['SpeciesConc_NO'] + ds['SpeciesConc_NO2']).isel(lev=0).squeeze()

    stepsize = 4

    ax = plt.axes(projection=ccrs.Mollweide())
    ax.set_global()

    ax.coastlines()
    print(da.min())
    print(da.max())
    norm = plt.Normalize(da.quantile(0.05).item(), da.quantile(0.95).item())
    for nf in range(6):
        X = grid.xe.isel(nf=nf).values
        Y = grid.ye.isel(nf=nf).values

        pcolormesh2(X, Y,  da.isel(nf=nf), 12, norm)



    plt.show()
    # from tqdm import tqdm
    #
    # def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
    #     xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    #     yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    #     for xm, ym in zip(xx_majors, yy_majors):
    #         ax.plot(xm, ym, transform=ccrs.PlateCarree(), color='k', linewidth=0.8, linestyle='-')
    #
    # import xarray as xr
    # from sg.grids import CubeSphere, StretchedGrid
    #
    # #ds = xr.open_dataset('/home/liam/analysis/ensemble-2/c180e.nc')
    # ds = xr.open_dataset('/extra-space/foobar/c180e-N-species.nc')
    # layer=35
    #
    # ID='EU1'
    #
    # da = ds['SpeciesConc_HNO3'].isel(lev=layer).sel(ID='CTL').squeeze()
    #
    # grid = CubeSphere(180)
    #
    #
    # plt.figure(figsize=(8.5,11))
    #
    # ax = plt.subplot(2,1,1, projection=ccrs.EqualEarth())
    # ax.coastlines()
    # ax.set_global()
    #
    # # norm = plt.Normalize(float(da.min()), float(da.max()))
    # # for face in tqdm(range(6)):
    # #     pcolormesh2(grid.xe(face), grid.ye(face), da.isel(face=face), 180 if face != 2 else 20, norm)
    #
    # ds_sg = ds.sel(ID=ID)
    # da_ctl = ds.sel(ID='CTL')
    # da = (ds_sg['SpeciesConc_HNO3'].isel(lev=layer).squeeze() - da_ctl['SpeciesConc_HNO3'].isel(lev=layer).squeeze())/da_ctl['SpeciesConc_HNO3'].isel(lev=layer).squeeze()
    # stretched_grid = StretchedGrid(int(ds_sg.cs_res), float(ds_sg.stretch_factor), float(ds_sg.target_lat), float(ds_sg.target_lon))
    #
    # draw_major_grid_boxes_naive(plt.gca(), stretched_grid.xe(5), stretched_grid.ye(5))
    # ax = plt.subplot(2,1,2, projection=ccrs.EqualEarth())
    # ax.coastlines()
    # ax.set_global()
    # norm = plt.Normalize(vmin=-0.5, vmax=0.5)
    # for face in tqdm(range(6)):
    #     pcolormesh2(grid.xe(face), grid.ye(face), da.isel(face=face), 180 if face != 2 else 20, norm, cmap='RdBu')
    # draw_major_grid_boxes_naive(plt.gca(), stretched_grid.xe(5), stretched_grid.ye(5))



    # plt.savefig(f'{ID}-{layer}-35.png', dpi=300)
    #plt.show()