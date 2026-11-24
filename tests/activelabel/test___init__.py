import re

import activelabel


def test_the_package_has_a_version_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", activelabel.__version__)
