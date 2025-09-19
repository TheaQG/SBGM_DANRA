"""
Utility functions for handling variable-specific operations such as unit conversions,
cropping to regions, and retrieving plotting specifications.
"""

import logging

# Setup logging
logger = logging.getLogger(__name__)

def get_units(cfg):
    """
        Get the specifications for plotting samples during training.
        Colors, labels, and other parameters are based on the configuration.
    """

    
    units = {"temp": r"$^\circ$C",
             "prcp": "mm",
             "cape": "J/kg",
             "nwvf": "m/s",
             "ewvf": "m/s",
             "msl": "hPa",
             "z_pl_250": "m",
             "z_pl_500": "m",
             "z_pl_850": "m",
             "z_pl_1000": "m",
             }


    hr_unit = units[cfg['highres']['variable']]
    lr_units = []
    for key in cfg['lowres']['condition_variables']:
        if key not in units:
            raise ValueError(f"Variable '{key}' not found in units dictionary.")
        else:
            lr_units.append(units[key])

    return hr_unit, lr_units

def get_unit_for_variable(variable: str):
    """
    Get the unit string for a specific variable.
    """
    units = {
        "temp": r"$^\circ$C",
        "prcp": "mm",
        "cape": "J/kg",
        "nwvf": "m/s",
        "ewvf": "m/s",
        "msl": "hPa",
        "z_pl_250": "m",
        "z_pl_500": "m",
        "z_pl_850": "m",
        "z_pl_1000": "m",
    }

    if variable not in units:
        raise ValueError(f"[get_unit_for_variable] Variable '{variable}' not found in units dictionary.")
    return units[variable]

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


def get_color_for_variable(variable: str, model: str):
    """
    Get a specific color for a variable based on the model type.
    Models can be DANRA or ERA5 - same variables have different colors for the two models.
    """
    if model.lower() == "danra":
        colors = {
            "temp": "cornflowerblue",
            "prcp": "darkorange",
            "cape": "forestgreen",
            "nwvf": "firebrick",
            "ewvf": "darkmagenta",
            "msl": "teal",
            "z_pl_250": "pink",
            "z_pl_500": "chocolate",
            "z_pl_850": "orange", 
            "z_pl_1000": "royalblue"
        }
    elif model.lower() == "era5":
        colors = {
            "temp": "mediumturquoise",
            "prcp": "goldenrod",
            "cape": "olive",
            "nwvf": "coral",
            "ewvf": "mediumpurple",
            "msl": "skyblue",
            "z_pl_250": "orchid",
            "z_pl_500": "coral",
            "z_pl_850": "tan",
            "z_pl_1000": "midnightblue"
        }
    else:
        # Default colors if model is unknown
        colors = {
            "temp": "cornflowerblue",
            "prcp": "darkorange",
            "cape": "forestgreen",
            "nwvf": "firebrick",
            "ewvf": "darkmagenta",
            "msl": "teal",
            "z_pl_250": "pink",
            "z_pl_500": "chocolate",
            "z_pl_850": "orange", 
            "z_pl_1000": "royalblue"
        }

    if variable not in colors:
        raise ValueError(f"[get_color_for_variable] Variable '{variable}' not found in color dictionary for model '{model}'.")
    
    return colors[variable]


def get_cmaps(cfg):
    """
        Get the colormaps for plotting samples during training.
        Colormaps are based on the configuration.
    """
    cmaps = {"temp": "plasma",
             "prcp": "inferno",
             "cape": "viridis",
             "nwvf": "cividis",
             "ewvf": "magma",
             "msl": "coolwarm",
             "z_pl_250": "coolwarm",
             "z_pl_500": "coolwarm",
             "z_pl_850": "coolwarm",
             "z_pl_1000": "coolwarm",
             }
    

    hr_cmap = cmaps[cfg['highres']['variable']]
    lr_cmaps = {}
    for key in cfg['lowres']['condition_variables']:
        if key not in cmaps:
            raise ValueError(f"Variable '{key}' not found in cmap dictionary.")
        else:
            lr_cmaps[key] = cmaps[key]

    return hr_cmap, lr_cmaps

def get_cmap_for_variable(variable: str):
    """
    Get the matplotlib colormap name for a specific variable.
    """
    cmaps = {"temp": "plasma",
             "prcp": "inferno",
             "cape": "viridis",
             "nwvf": "cividis",
             "ewvf": "magma",
             "msl": "coolwarm",
             "z_pl_250": "coolwarm",
             "z_pl_500": "coolwarm",
             "z_pl_850": "coolwarm",
             "z_pl_1000": "coolwarm",
             }

    if variable not in cmaps:
        # If variable not found, return a default colormap
        logger.warning(f"[get_cmap_for_variable] Variable '{variable}' not found in cmap dictionary. Using default 'viridis'.")
        return "viridis"
    return cmaps[variable]