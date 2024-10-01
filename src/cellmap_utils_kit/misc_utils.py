"""
Utility functions for miscellaneous operations.

This module provides utility functions for general-purpose tasks that may be
used across various parts of the codebase, such as extracting specific patterns
from strings.

Key Functions:
-------------
- `extract_crop_name(path: str) -> str | None`:
  Extracts the crop name from a given path string. It looks for the first
  occurrence of the pattern "crop" followed by a number and returns that name.

Dependencies:
------------
- re: For regular expression operations.

Usage:
-----
1. Import the `extract_crop_name` function from this module.
2. Call the function with a string path to extract the crop name.

Example:
-------
```python
crop_name = extract_crop_name("/path/to/crop12.zarr/data")
print(crop_name)  # Output: crop12
```

"""

import re


def extract_crop_name(path: str) -> str | None:
    """Extract the crop name from a path.

    From a string takes the first occurence of "crop"+some number and returns that.

    Args:
        path (str): Path from which to extract crop name

    Returns:
        str | None: Name of the crop. If no crop name was found returns None.

    """
    match = re.search(r"crop\d+", path)
    return match.group(0) if match else None
