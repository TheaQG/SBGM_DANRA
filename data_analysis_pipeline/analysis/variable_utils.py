from sbgm.utils import get_units, get_cmaps

def correct_variable_units(var_name, data):
    
    """
    Apply basic unit corrections to known variables.
    E.g., convert temperature from K to C, precipitation from m to mm.
    """
    if var_name in ["temp", "t2m"]:
        data = data - 273.15
    elif var_name in ["prcp"]:
        data = data * 1000  # from m to mm
    return data

def crop_to_region(data, crop_region):
    """
    Crop the data to a specific subregion: [x_start, x_end, y_start, y_end].
    """
    x_start, x_end, y_start, y_end = crop_region
    return data[..., y_start:y_end, x_start:x_end]

def get_var_name_short(varname):
    """
    Optionally standardize variable naming (e.g., aliasing or shortening).
    """
    aliases = {
        "t2m_ave": "temp",
        "tp_tot": "prcp"
    }
    return aliases.get(varname, varname)

