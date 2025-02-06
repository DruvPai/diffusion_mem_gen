import os
import yaml
from typing import Callable, Dict, List, Optional, Union, Any
import wandb
from wandb.apis.public.runs import Run
from wandb.apis.public.files import File
from wandb import Api
from pathlib import Path
import dill as pickle


def download_files_from_run(
    run: Run,
    local_dir: Union[str, Path],
    filter_fn: Callable[[File], bool] | None = None,
    force_download: bool = False,
) -> List[Path]:
    """
    Download files from a wandb run based on a filter function.
    
    Args:
        run: A wandb Run object.
        local_dir: Local directory to download files to.
        filter_fn: A function that takes a wandb File object and returns
            True if the file should be downloaded.
        force_download: If True, download even if the file exists locally.
    
    Returns:
        List of paths to downloaded files.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    for file in run.files():
        if filter_fn is not None and not filter_fn(file):
            continue
        
        local_path = local_dir / file.name
        if local_path.exists() and not force_download:
            print(f"File {file.name} already exists at {local_path}. Skipping download.")
            downloaded_files.append(local_path)
            continue
        
        file.download(root=str(local_dir), replace=True)
        print(f"Downloaded {file.name} to {local_path}")
        downloaded_files.append(local_path)
    
    return downloaded_files


def download_files_from_runs(
    api: Api | None = None,
    entity: str | None = None,
    project: str = "default",
    run_filter_fn: Callable[[Run], bool] | None = None,
    file_filter_fn: Callable[[File], bool] | None = None,
    output_dir: Union[str, Path] = "wandb_downloads",
    group: str | None = None,
    force_download: bool = False,
) -> Dict[str, List[Path]]:
    """
    Download files from multiple wandb runs based on filter functions.
    
    Args:
        api: A wandb Api object. If None, a new one will be created.
        entity: The wandb entity (user or team). If None, defaults to your username.
        project: The wandb project. Required.
        run_filter_fn: A function that takes a wandb Run object and returns
            True if files from this run should be downloaded.
        file_filter_fn: A function that takes a wandb File object and returns
            True if the file should be downloaded.
        output_dir: Base directory for downloads.
        group: If provided, only runs in this group will be considered.
        force_download: If True, download even if files exist locally.
    
    Returns:
        Dictionary mapping run IDs to lists of downloaded file paths.
    """
    api = api or wandb.Api()
    
    # Build the query
    filters = {}
    if group is not None:
        filters["group"] = group
    
    # Get runs
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    
    # Download files from each run
    result = {}
    for run in runs:
        if run_filter_fn is not None and not run_filter_fn(run):
            continue
        
        run_dir = Path(output_dir)
        if entity:
            run_dir = run_dir / entity
        run_dir = run_dir / project / run.name

        downloaded = download_files_from_run(
            run=run,
            local_dir=run_dir,
            filter_fn=file_filter_fn,
            force_download=force_download,
        )
        
        if downloaded:
            result[run.id] = downloaded
    
    return result


def read_pkl(pkl_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a results.pkl file into a Python dictionary.
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def read_wandb_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a wandb config.yaml file into a Python dictionary.
    
    Args:
        config_path: Path to the config.yaml file.
    
    Returns:
        Dictionary representation of the config.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

import wandb

def check_run_exists(api: Api | None, entity: str, project: str, run_name: str) -> bool:
    """
    Checks if a wandb run with a specific name exists in a project.

    Args:
        api: A wandb Api object. If None, a new one will be created.
        entity: The wandb entity (user or team).
        project: The wandb project name.
        run_name: The display name of the run to check for.

    Returns:
        True if a run with that name exists, False otherwise.
    """
    if not api:
        api = wandb.Api()
    try:
        # Construct the path to the project
        project_path = f"{entity}/{project}"
        # List runs in the project, filtering by the display_name
        # The 'display_name' field is typically what's shown as the run name in the UI
        runs = api.runs(path=project_path, filters={"display_name": run_name})

        # If the list of runs is not empty, at least one run with that name exists
        return len(runs) > 0
    except wandb.Error as e:
        # This might occur if the project itself doesn't exist or there's a communication issue
        print(f"Communication error with wandb: {e}")
        return False
    except Exception as e:
        # Handle other potential exceptions
        print(f"An unexpected error occurred: {e}")
        return False