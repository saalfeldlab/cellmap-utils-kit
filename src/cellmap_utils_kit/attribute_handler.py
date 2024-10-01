"""
cellmap-utils.kit.attribute_handler: Utility functions to deal with cellmap
    style attributes.
"""


def add_scalelevel_to_attributes(
    attrs_as_dict: dict, scalelvl: str, scale: list[float], translation: list[float]
) -> dict:
    """Add an additional scale level to attributes describing a multiscale
    pyramid.

    Args:
        attrs_as_dict (dict): Attributes dictionary with multiscales.
        scalelvl (str): The name/path of the scale level that should be added.
        scale (list[float]): The scale factor for the added scale level
        translation (list[float]): The translation for the added scale level.

    Returns:
        dict: Updated attributes dictionary with new scale level added.

    """
    dataset_list = attrs_as_dict["multiscales"][0]["datasets"]
    dataset_list.append(
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": translation},
            ],
            "path": scalelvl,
        }
    )
    attrs_as_dict["multiscales"][0]["datasets"] = dataset_list
    return attrs_as_dict


def initialize_multiscale_attributes() -> dict:
    """Return a dictionary representing an empty mutliscale pyramid.

    Returns:
        dict: Dictionary representing an empty multiscale pyramid that can be
            used to initialize attributes.

    """
    axes = [{"name": axname, "type": "space", "unit": "nanometer"} for axname in "zyx"]
    ct = [{"scale": [1.0, 1.0, 1.0], "type": "scale"}]
    attrs = {
        "multiscales": [{"axes": axes, "coordinateTransformations": ct, "datasets": []}]
    }
    return attrs


def extract_single_scale_attrs(attrs_as_dict: dict, scalelvl: str, rename: str) -> dict:
    """Turn attributes describing a multiscale pyramid into a multiscale
    pyramid that just contains a single scale level.

    Args:
        attrs_as_dict (dict): Attributes describing multiscale pyramid.
        scalelvl (str): Name/path of scale level that should be extracted
        rename (str): Name/path for the single scale level in the reduced
            attributes.

    Raises:
        ValueError: If scale level `scalelvl` is not referenced in `attrs_as_dict`.

    Returns:
        dict: Dictionary describing a multiscale pyramid, but now only for a
        single scale level.

    """
    single_attr = next(
        (
            ms
            for ms in attrs_as_dict["multiscales"][0]["datasets"]
            if ms["path"] == scalelvl
        ),
        None,
    )
    if single_attr is not None:
        attrs_as_dict["multiscales"][0]["datasets"] = [single_attr]
    else:
        msg = f"Scale level {scalelvl} not found in attributes."
        raise ValueError(msg)

    if scalelvl != rename:
        attrs_as_dict["multiscales"][0]["datasets"][0]["path"] = rename
    return attrs_as_dict


def get_scale_and_translation(
    attrs_as_dict: dict, scalelvl: str
) -> tuple[list[float], list[float]]:
    """Infer scale and translation of scale level `scalelvl` from dictionary describing
    multiscale attributes `attrs_as_dict`.

    Args:
        attrs_as_dict (dict): Dictionary describing multiscale attributes.
        scalelvl (str): Name of scale level for which to get scale and translation.

    Raises:
        ValueError: If the specified scale level does not exist in the multiscale
            attributes.
        ValueError: If an unknown coordinate transformation type is encountered.
        ValueError: If the translation or scale value is not found for the specified
            scale level.

    Returns:
        tuple[list[float], list[float]]: Two lists representing the scale values and
            translation values, respectively.

    """
    for ms in attrs_as_dict["multiscales"][0]["datasets"]:
        if ms["path"] == scalelvl:
            scale = None
            translation = None
            for ct in ms["coordinateTransformations"]:
                if ct["type"] == "scale":
                    scale = ct["scale"]
                elif ct["type"] == "translation":
                    translation = ct["translation"]
                else:
                    msg = (
                        f"Unknown coordinate transformation type in attributes for "
                        f"{scalelvl}: {ct['type']}"
                    )
                    raise ValueError(msg)
            if translation is None or scale is None:
                msg = (
                    f"Did not find translation and scale value in attributes for "
                    f"{scalelvl}: {translation=}, {scale=}"
                )
                raise ValueError(msg)
            return scale, translation
    msg = f"Did not find attributes for {scalelvl} in {attrs_as_dict}"
    raise ValueError(msg)
