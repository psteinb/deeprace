# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from subprocess import check_output
from verbs.utils import uuid_from_this, yaml_this
import yaml

def test_create_uuid_diff_input():

    hash1 = uuid_from_this("foo", "bar", 12, 42)
    hash2 = uuid_from_this("foo", "bar", 12, 43)

    assert hash1 != hash2

def test_create_uuid_diff_time():

    hash1 = uuid_from_this("foo", "bar", 12, 42)
    hash2 = uuid_from_this("foo", "bar", 12, 42)

    assert hash1 != hash2


def test_yaml_from_kwargs():

    yaml_str = yaml_this(check="this",out=42)

    reloaded = yaml.load(yaml_str)

    assert yaml_str.count("check")
    assert yaml_str.count("this")
    assert yaml_str.count("42")
