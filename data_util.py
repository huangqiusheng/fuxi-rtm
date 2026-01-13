import numpy as np
import xarray as xr

__all__ = ["chunk", "print_xarray", "save_with_progress"]


def chunk(ds, time=1, **kwargs):
    if isinstance(ds, xr.DataArray):
        dims = {k: v for k, v in zip(ds.dims, ds.shape)}
    else:
        dims = {k: v for k, v in ds.sizes.items()}

    if "time" in dims:
        dims["time"] = time

    for k, v in kwargs.items():
        if k in dims:
            dims[k] = v

    ds = ds.chunk(dims)
    return ds


def print_xarray(ds, desp="", names=[]):

    if isinstance(ds, xr.Dataset):
        for v in ds:
            print_xarray(ds[v], desp=desp, names=names)
        return
    
    v = ds
    msg = f"{v.name.upper()} {desp}: \nshape: \033[94m{v.shape}\033[0m"

    if "time" in ds.dims:
        start_date = ds.time[0].dt.date.item()
        end_date = ds.time[-1].dt.date.item()
        msg += f", time: \033[94m{len(ds.time)} = ({start_date} ~ {end_date})\033[0m"

    if "lat" in ds.dims and "lon" in ds.dims:
        lat = ds.lat.values
        lon = ds.lon.values
        msg += f", latlon: \033[94m({lat[0]:.3f} ~ {lat[-1]:.3f}) x ({lon[0]:.3f} ~ {lon[-1]:.3f})\033[0m"

    if "level" in v.dims:
        if len(names) > 0:
            v = v.sel(level=np.intersect1d(names, v.level))
        for lvl in v.level.data:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: \033[91m{x.min():.3f} ~ {x.max():.3f}\033[0m"
    elif "depth" in v.dims:
        for lvl in v.depth.data:
            x = v.sel(depth=lvl).values
            msg += f"\ndepth: {lvl:.2f}, value: \033[91m{x.min():.3f} ~ {x.max():.3f}\033[0m"
    elif "channel" in v.dims:
        if len(names) > 0:
            v = v.sel(channel=np.intersect1d(names, v.channel))        
        for ch in v.channel.data:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: \033[91m{x.min():.3f} ~ {x.max():.3f}\033[0m"            
    else:
        x = v.values
        msg += f", value: \033[91m{x.min():.3f} ~ {x.max():.3f}\033[0m"

    print(msg)



def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
    
    ds = ds.astype(dtype)
    if 'channel' in ds.coords and ds.coords['channel'].dtype == 'object':
        ds = ds.assign_coords(
            channel=ds.coords['channel'].astype('U50')  # 'U50' 表示最大50字符的Unicode字符串
        )

    if save_name.endswith(".nc"):
        obj = ds.to_netcdf(save_name, compute=False)
    elif save_name.endswith(".zarr"):
        obj = ds.to_zarr(save_name, compute=False)
    else:
        raise ValueError("save_type must be 'nc' or 'zarr'!")
    
    with ProgressBar():
        obj.compute()



