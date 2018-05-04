#thanks to https://stackoverflow.com/a/11887825
def versiontuple(v, version_index = -1):
    """ convert a version string to a tuple of integers
    argument <v> is the version string, <version_index> refers o how many '.' splitted shall be returned

    example:
    versiontuple("1.7.0") -> tuple([1,7,0])
    versiontuple("1.7.0", 2) -> tuple([1,7])
    versiontuple("1.7.0", 1) -> tuple([1])

    """

    temp_list = map(int, (v.split(".")))
    return tuple(temp_list)[:version_index]
