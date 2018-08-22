import pytest
import os
import glob
import sys
import numpy as np

print(sys.path)
from tifffile import imread
from datasets.care_denoise2d_resave import resave_to_chunks, load_chunk
from datasets.care_denoise2d_data import create_data, create_data_from_chunks

@pytest.fixture(scope="module")
def location():
    folder = '/projects/steinbac/care-denoise2d'
    stem = 'pytest-care-denoise2d'

    def fin():
        found = glob.glob(os.path.join(folder,stem+".np*"))
        if len(found) > 0:
            for i in found:
                os.remove(i)
    return {"dir" : folder, "stem" : stem}



def test_chunk_resave_produces_nonzero(location):

    chunkloc = resave_to_chunks(root=location["dir"],
                                n_imgs=10,
                                output_stem=location["stem"])

    assert os.path.exists(chunkloc)
    assert os.stat(chunkloc).st_size > 0

def test_correct_image_count(location):

    chunkloc = resave_to_chunks(root=location["dir"],
                                n_imgs=10,
                                output_stem=location["stem"])

    loaded = np.load(chunkloc)
    assert len(loaded.files) > 0
    assert len(loaded.files) == 10

def test_correct_image_size(location):
    """ against output of tiffinfo

    TIFF Directory at offset 0x8 (8)
  Subfile Type: (0 = 0x0)
  Image Width: 696 Image Length: 520
  Bits/Sample: 16
  Sample Format: unsigned integer
  Compression Scheme: None
  Photometric Interpretation: min-is-black
  Samples/Pixel: 1
  Rows/Strip: 520
  Planar Configuration: single image plane
  ImageDescription: {"shape": [520, 696]}
  Software: tifffile.py
  DateTime: 2018:02:16 18:01:13
"""
    chunkloc = resave_to_chunks(root=location["dir"],
                                n_imgs=10,
                                output_stem=location["stem"])

    loaded = np.load(chunkloc)
    assert len(loaded.files) > 0

    first = loaded[loaded.files[0]]
    assert first.shape != ()
    assert first.shape == (520,696)

def test_compare_image_with_tifffile(location):

    chunkloc = resave_to_chunks(root=location["dir"],
                                n_imgs=10,
                                output_stem=location["stem"])

    imgs = load_chunk(chunkloc)

    assert len(imgs) > 0
    first = imgs[0]

    assert first.shape == (520,696)

    fnames = glob.glob(os.path.join(location["dir"],"*tif"))
    first_tiff = imread(fnames[0])

    assert first.shape == first_tiff.shape

    for i in imgs:
        assert i.shape == first_tiff.shape

def test_chunk_integration(location):

    expected = create_data(root=location["dir"], n_imgs=10)

    chunkloc = resave_to_chunks(root=location["dir"],
                                n_imgs=10,
                                output_stem=location["stem"])

    observed = create_data_from_chunks(chunk_loc=chunkloc,n_imgs=10)

    for i in range(len(expected)):
        assert expected[i].dtype == observed[i].dtype
        assert expected[i].shape == observed[i].shape
        assert np.all( np.abs(expected[i] - observed[i]) < 1e5 )
