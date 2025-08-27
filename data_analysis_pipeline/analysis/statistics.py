import datetime
from importlib_metadata import files
import numpy as np

def compute_statistics(data, aggregate=False, agg_time="daily", agg_method="mean"):
    """
        Compute statistics for the given data.
        Mean, std, min, max etc. per file or full stack
        If aggregate = True, aggregate temporally before computing statistcs
    """
    cutouts = data["cutouts"]
    if aggregate: 
        aggregation = aggregate_data(data, agg_time, agg_method)
        stack = aggregation["stack"]
        cutouts = aggregation["cutouts"]
    else:
        stack = np.stack([x.values for x in cutouts]) # Shape: (T, H, W)

    # Compute basic statistics
    stats = {
        "mean": np.mean(stack, axis=0),
        "std": np.std(stack, axis=0),
        "min": np.min(stack, axis=0),
        "max": np.max(stack, axis=0),
        "median": np.median(stack, axis=0),
        "percentile_25": np.percentile(stack, 25, axis=0),
        "percentile_75": np.percentile(stack, 75, axis=0)
    }

    return stats


def aggregate_data(data, agg_time, agg_method):
    """ 
        Aggregate data across multiple files or data chunks.
        This could involve averaging, summing, or otherwise combining the data.
    """
    aggregation_time = agg_time  # e.g., "weekly", "monthly", "yearly"
    
    cutouts = data["cutouts"]
    timestamps = data["timestamps"]

    # Convert timestamps to datetime objects if not already
    if not isinstance(timestamps[0], datetime.datetime):
        timestamps = [datetime.datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]

    # Group indices by aggregation_time
    groups = {}
    for idx, ts in enumerate(timestamps):
        if aggregation_time == "weekly":
            key = (ts.year, ts.isocalendar()[1])  # Year and week number
        elif aggregation_time == "monthly":
            key = (ts.year, ts.month)  # Year and month
        elif aggregation_time == "yearly":
            key = (ts.year,)  # Year only
        else:
            raise ValueError(f"Unsupported aggregation_time: {aggregation_time}")

        # Store the index in the appropriate group
        groups.setdefault(key, []).append(idx)

    aggregated_cutouts = []
    aggregated_stack = []
    aggregated_timestamps = []
    for key, indices in groups.items():
        # Group cutouts by the current key
        group_cutouts = [cutouts[i] for i in indices]
        # Stack the group cutouts into list of arrays
        group_arrays = [c.values for c in group_cutouts]
        # Stack the group arrays into a single array
        stack_group = np.stack(group_arrays)  # Shape: (num_in_group, H, W)

        if agg_method == "mean":
            # Compute mean across time axis (axis = 0)
            agg_cutouts = group_cutouts[0].copy(data=np.mean(stack_group, axis=0))
            
        elif agg_method == "sum":
            # Compute sum across time axis
            agg_cutouts = group_cutouts[0].copy(data=np.sum(stack_group, axis=0))
            
        elif agg_method == "max":
            # Compute max across time axis
            agg_cutouts = group_cutouts[0].copy(data=np.max(stack_group, axis=0))

        elif agg_method == "min":
            # Compute min across time axis
            agg_cutouts = group_cutouts[0].copy(data=np.min(stack_group, axis=0))

        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

        aggregated_cutouts.append(agg_cutouts)
        aggregated_stack.append(agg_cutouts.values)

        # Generate representative timestamp for the group (always start of _)
        if aggregation_time == "weekly":
            year, week = key
            dt = datetime.datetime(year, 1, 1) + datetime.timedelta(weeks=week-1) # Start of the week
        elif aggregation_time == "monthly":
            year, month = key
            dt = datetime.datetime(year, month, 1) # Start of the month
        elif aggregation_time == "yearly":
            (year, ) = key[0]
            dt = datetime.datetime(year, 1, 1) # Start of the year
        else:
            raise ValueError(f"Unsupported aggregation_time: {aggregation_time}")

        aggregated_timestamps.append(dt)

    # Stack the aggregated cutouts and return
    return {
        "cutouts": aggregated_cutouts,
        "stack": aggregated_stack,
        "timestamps": aggregated_timestamps
    }
