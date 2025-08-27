import os

def build_data_path(base_dir, model_type, variable, domain_size, split=None, zarr=False):
    """
    Constructs the full path to the data directory.

    Args:
        base_dir (str): Root data directory, e.g., ${DATA_DIR}
        model_type (str): e.g., 'ERA5' or 'DANRA'
        variable (str): e.g., 'prcp'
        domain_size (list or tuple): e.g., [589, 789]

    Returns:
        str: Full path to data directory.
    """
    size_str = f"{domain_size[0]}x{domain_size[1]}"
    path = os.path.join(
        base_dir,
        f"data_{model_type}",
        f"size_{size_str}",
        f"{variable}_{size_str}"
    )

    if split == 'all':
        path = os.path.join(path, 'all')
    elif split in ['train', 'val', 'test']:
        if zarr:
            path = os.path.join(path, 'zarr_files', split + '.zarr')
        else:
            path = os.path.join(path, split)

    return path