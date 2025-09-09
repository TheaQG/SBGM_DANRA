from sbgm.utils import get_units, get_cmaps

def correct_variable_units(var_name, model, data):
    
    """
    Apply basic unit corrections to known variables.
    E.g., convert temperature from K to C, precipitation from m to mm.
    """
    if var_name in ["temp", "t2m"]:
        data = data - 273.15
    elif var_name in ["prcp", "tp"] and model in ["DANRA"]:
        # Make sure no negative values (set <0 to 1e-10)
        data[data < 0] = 1e-10
    elif var_name in ["prcp"] and model in ["ERA5"]:
        data = data * 1000  # from m to mm
        # Make sure no negative values after conversion (set <0 to 1e-10)
        data[data < 0] = 1e-10
    elif var_name in ["cape"] and model in ["ERA5"]:
        data = data / 1000  # from J/kg to kJ/kg
        # Also ensure no negative CAPE values
        data[data < 0] = 1e-10
    elif var_name in ["msl"] and model in ["ERA5"]:
        data = data / 100  # from Pa to hPa
    elif var_name in ["pev"] and model in ["ERA5"]:
        data = data / 1000  # from Pa to hPa
    elif var_name in ["z_pl_1000", "z_pl_250", "z_pl_500", "z_pl_850"] and model in ["ERA5"]:
        data = data / 9.81  # from geopotential to geopotential height in meters
        
    return data

def crop_to_region(data, crop_region):
    """
    Crop the data to a specific subregion: [x_start, x_end, y_start, y_end].
    """
    [x_start, x_end, y_start, y_end] = crop_region
    return data[x_start:x_end, y_start:y_end]

def get_var_name_short(varname, model, domain_size=[589, 789]):
    """
    Optionally standardize variable naming (e.g., aliasing or shortening).
    """
    domain_size_str = f"{domain_size[0]}x{domain_size[1]}"

    if model == 'DANRA':
        aliases = {
            "temp": "t2m_ave",
            "prcp": "tp_tot"
        }
    elif model == 'ERA5':
        aliases = {
            "cape": f"cape_{domain_size_str}",
            "ewvf": f"wvf_east_{domain_size_str}",
            "msl": f"msl_{domain_size_str}",
            "nwvf": f"wvf_north_{domain_size_str}",
            "pev": f"pev_{domain_size_str}",
            "prcp": f"tp_{domain_size_str}",
            "temp": f"t2m_{domain_size_str}",
            "z_pl_1000": f"z_pl_1000_hPa_{domain_size_str}",
            "z_pl_250": f"z_pl_250_hPa_{domain_size_str}",
            "z_pl_500": f"z_pl_500_hPa_{domain_size_str}",
            "z_pl_850": f"z_pl_850_hPa_{domain_size_str}"
        }
    else:
        aliases = {}
    return aliases.get(varname, varname)

