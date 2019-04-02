import os
import logging
import re
import importlib

try:
    from importlib import util as ilib_util
except BaseException:
    raise
else:
    finder = ilib_util.find_spec


def import_model(name):
    """ import a model and return the imported symbol """

    expected_models_dir = os.path.dirname(os.path.abspath(__file__))
    expected_models_dir = os.path.join(os.path.dirname(expected_models_dir), "models")

    if not os.path.exists(expected_models_dir):
        logging.warning("%s was not found in %s", expected_models_dir, os.curdir)
        return None

    expected_location = os.path.join(expected_models_dir, name + ".py")
    if not os.path.exists(expected_location):
        logging.warning("%s was not found in %s", expected_location, os.curdir)
        return None
    else:
        logging.info("found %s implementation", name)

    try_this = ["", ".", ".."]
    full_name = "models.%s" % (name)

    ld = None
    for it in try_this:
        ld = finder(full_name, package="%smodels" % it)

        if ld:
            break
        else:
            logging.info("no finder with %s at %8s (cwd: %s)", full_name, it, os.path.abspath(os.curdir))

    if ld is None:
        ld = finder(name, path="models")

    if ld is None:
        return None

    return importlib.import_module(full_name)


def load_model(descriptor):
    """ given a string, return a tuple of the respective module loader and a dictionary of constant parameters """

    model_id_candidates = re.findall(r'^[^(\d)]+', descriptor)

    if not model_id_candidates:
        return None

    model_id = model_id_candidates[0].rstrip("_")
    logging.info("importing %s (from %s)", model_id, descriptor)
    loaded = import_model(model_id)
    if not loaded:
        logging.error("unable to load %s inferred from %s", model_id, descriptor)
        return None

    param_dict = loaded.params_from_name(descriptor)
    return (loaded, param_dict)
