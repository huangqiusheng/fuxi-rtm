import argparse
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from time import perf_counter
from copy import deepcopy
from data_util import *
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="exps/RTM_onnx/base_tf/model/test.onnx", # base_tf exp_re4
                    help="Path to the ONNX file for the FuXi-S2S model.")
parser.add_argument('--input', type=str, default="datasets/era5.rtm.02_25.6h.c109.new3/", 
                    help="Path to the input NetCDF data file.")
parser.add_argument('--save_dir', type=str, default="eval/save", 
                    help="Directory where the prediction output will be saved.")
parser.add_argument('--save_type', type=str, default="zarr", choices=["nc", "zarr"])
parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument('--dtype', type=str, default="fp32", choices=["fp16", "fp32"])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--total_step', type=int, default=400) # 40
parser.add_argument('--total_member', type=int, default=1)
parser.add_argument('--hour_interval', type=int, default=6)
parser.add_argument('--time_splite', type=str, nargs="+", default=["20240101", "20250531"])
parser.add_argument('--init_time_hour', type=str, nargs="+", default=[0, 12])
args = parser.parse_args()
ill_time=[pd.Timestamp('202501010000')]
ill_time+=[pd.Timestamp('202401010000')]
LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVELS_12 = [50, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def extract_letters(s):
    match = re.match(r'([a-zA-Z]+)\d+', s)
    if match:
        return match.group(1)
    return s 

def save_pred(output, input, steps, members):
    save_type = args.save_type
    save_dir = args.save_dir
    init_time = pd.to_datetime(input.time.data[-1])
    tmp_dir = os.path.join(save_dir, init_time.strftime("%Y%m%d-%H"))
    os.makedirs(tmp_dir, exist_ok=True)

    pred = xr.DataArray(
        name="output",
        data=output[None], # m t c h w
        dims=['time', 'member', 'step', 'channel', 'lat', 'lon'],
        coords=dict(
            time=[init_time],
            member=members,
            step=steps,
            channel=input.channel,
            lat=input.lat,
            lon=input.lon,
        )
    ).astype(np.float32)
    
    save_name = os.path.join(tmp_dir, f'{steps[0]:03d}.{save_type}')
    save_with_progress(pred, save_name)


def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = os.cpu_count() or 1
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(model_name, sess_options=options, providers=providers)
    return session



def run_inference(model, input, id_1=[], id_2=[], id_3=[]):
    total_step = args.total_step
    total_member = args.total_member
    hour_interval = args.hour_interval
    batch_size = args.batch_size
    input_names = [x.name for x in model.get_inputs()]

    dtype = dict(
        fp16=np.float16,
        fp32=np.float32,
    )[args.dtype]
    
    assert total_member % batch_size == 0, "total_member must be divisible by batch_size"   

    lat = input.lat.values 
    hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(hours=hour_interval)
    assert lat[0] == 90 and lat[-1] == -90
    print(f"\nInference process at {init_time} ...")
    
    for m in range(0, total_member, batch_size):
        first_time = perf_counter()
        members = m+1+np.arange(batch_size)
        print(f'\nInference for members {members} ...')
        
        new_input = np.repeat(input.values[None], batch_size, axis=0).astype(dtype)

        for t in range(total_step):
            valid_time = init_time + pd.Timedelta(hours=t * hour_interval)
            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([t] * batch_size, dtype=dtype)

            if "hour" in input_names:
                hour = [valid_time.hour/24] * batch_size
                inputs['hour'] = np.array(hour, dtype=dtype)     

            if "doy" in input_names:
                doy = min(365, valid_time.day_of_year)/365 
                inputs['doy'] = np.array([doy] * batch_size, dtype=dtype)

            start_time = perf_counter()
            new_input, = model.run(None, inputs)
            output = deepcopy(new_input[:, -1:])
            
            k = t * output.shape[1]
            steps = k+1+np.arange(output.shape[1])
            save_pred(output, input, steps, members)
            elapsed_time = perf_counter() - start_time
            
            print(f"members: {members}, step {t+1:03d}, time: {elapsed_time:.3f} secs")
            
            if t >= total_step:
                break

        total_time = perf_counter() - first_time
        print(f'Inference for members {members} done, take {total_time:.3f} secs')

    print(f"\nInference process at {init_time} done")

if __name__ == "__main__":
    assert os.path.exists(args.input), f"Input file {args.input} not found!"
    assert os.path.exists(args.model), f"Model file {args.model} not found!"

    engine = "zarr" if args.input.endswith("zarr") else "netcdf4"
    input = xr.open_zarr(args.input).sel(time=slice(*args.time_splite))
    # innorm
    use_zzjNorm = True
    m = xr.open_dataarray(os.path.join(args.input, 'mean.nc'))
    s = xr.open_dataarray(os.path.join(args.input, 'std.nc'))
    input = input*s+m 
    da = input.data
    ds = da.to_dataset(dim="channel")         
    if use_zzjNorm:
        ds["clwc200"] = (10**(ds["clwc200"] - 14) - 1e-17).clip(0, 1000)

        CLWC_LR = ["clwc"+str(i) for i in LEVELS_12]
        for var in CLWC_LR:
            ds[var] = (np.exp(ds[var] - 3) - 1e-3).clip(0, 1000)

        OTHER = ["tcc", "lcc", "mcc", "hcc"] + [f"cc{i}" for i in LEVELS_13]
        for var in OTHER:
            ds[var] = (np.exp(ds[var] - 3) - 1e-3).clip(0, 1000)

    LOG = ["tp"]
    for var in LOG:
        ds[var] = np.exp(ds[var].clip(0, 7)) - 1

    input = ds.to_array(dim="channel").sel(channel=da.channel).transpose("time", 'channel', "lat", "lon")

    print(f'Load FuXi ...')       
    start = perf_counter()
    model = load_model(args.model, args.device)
    print(f'Load FuXi take {perf_counter() - start:.2f} secs')
    
    time_all = input.time.values

    for t in time_all:
        if pd.Timestamp(t) - pd.Timedelta(hours=args.hour_interval) not in input.time.values:
            continue
        if pd.Timestamp(t).hour not in args.init_time_hour:
            continue
        if pd.Timestamp(t) in ill_time:
            continue
        input_cur = input.sel(time=[pd.Timestamp(t) - pd.Timedelta(hours=args.hour_interval), pd.Timestamp(t)]) # 2 c h w
        run_inference(model, input_cur)
