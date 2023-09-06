"""
This file has the following functionality:
    - download the archive files
    - verify the md5sums of the files downloaded
    - extract the archive
    - convert any files (if necessary)
    - parse the dataset (return images and classes)

This file should be COMPLETELY SELF-CONTAINED. This means that all necessary URLs, md5sums,
special rules, functions you add etc. should be contained within the text of this file.

The file ~does not~ assume that you are running python inside the root of the dataset folder,
instead all of the fixed API functions required the parameter dataset_root_path as input.

The file is divided into 4 sections to make it easy to review.
    0. dataset notes (note any issues found during dataset preparation)
    1. imports and constants (any special lists, dicts, etc)
    2. functions with fixed APIs (the function signatures are NOT allowed to be changed)
    3. dataset-specific helper functions
"""

#
# 0. notes and constants
#


# 1) copy the .h5 file into original_files and extract its content to converted_files using h5py
# 2) class_name found in https://astronn.readthedocs.io/en/latest/galaxy10.html and hard-coded in class_dict


#
# 1. constants
#

# only 1 file is necessary for this example dataset
DATASET_URL_1 = "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
DATASET_FILE_1 = "Galaxy10_DECals.h5"
DATASET_SUM_1 = "c6b7b4db82b3a5d63d6a7e3e5249b51c"


#
# 1. imports
#


import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from dataset_utils import (
    VALID_IMAGE_EXTENSIONS,
    DatasetImage,
    chdir_with_create,
    create_images_database,
    filter_file_list,
    get_all_files_in_directory,
    get_unique_directories_from_files_list,
    is_image_valid,
    run_cmd_get_output,
)
from PIL import Image

#
# 2. functions with fixed APIs
#


def download_dataset(dataset_root_path: str):
    """This function:
    - downloads the necessary archive and supporting files (if possible)
    - puts them in ./archives/
    """
    print("download dataset")

    # change dir
    # chdir_with_create(dataset_root_path + "archives/")

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again
    # if not os.path.isfile(DATASET_FILE_1):
    #    run_cmd_get_output(["wget", DATASET_URL_1])


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    # verify md5sums for each downloaded file
    md5sum_result_1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]

    # the return should only be True if all md5sums match
    return md5sum_result_1 == DATASET_SUM_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("no need to extract dataset")


def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """

    print("convert dataset")
    chdir_with_create(dataset_root_path + "original_files/")
    run_cmd_get_output(
        [
            "cp",
            dataset_root_path + "archives/Galaxy10_DECals.h5",
            dataset_root_path + "original_files/",
        ]
    )
    f = h5py.File(dataset_root_path + "original_files/Galaxy10_DECals.h5", "r")
    images = f["images"]
    labels = np.array(f["ans"]).astype(np.int)

    chdir_with_create(dataset_root_path + "converted_files/")
    for i, img in enumerate(images):
        img = Image.fromarray(img.astype("uint8"), "RGB")
        img.save(f"{i}_{labels[i]}.png", "png")


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    files_list = get_all_files_in_directory(
        dataset_root_path + "converted_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # generate the class_dict by hard-coding
    class_dict.update({"Disturbed Galaxies": 0})
    class_dict.update({"Merging Galaxies": 1})
    class_dict.update({"Round Smooth Galaxies": 2})
    class_dict.update({"In-between Round Smooth Galaxies": 3})
    class_dict.update({"Cigar Shaped Smooth Galaxies": 4})
    class_dict.update({"Barred Spiral Galaxies": 5})
    class_dict.update({"Unbarred Tight Spiral Galaxies": 6})
    class_dict.update({"Unbarred Loose Spiral Galaxies": 7})
    class_dict.update({"Edge-on Galaxies without Bulge": 8})
    class_dict.update({"Edge-on Galaxies with Bulge": 9})

    label_class_dict = {v: k for k, v in class_dict.items()}
    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if problem_value != 0:
            # if any problems, set class id to -1
            class_int = -1
            set_value = -1
        else:
            class_from_path = image_path.split("/")[-1]
            class_from_path = class_from_path.split(".")[0]
            class_from_path = int(class_from_path.split("_")[-1])
            assert class_from_path in label_class_dict
            class_int = class_from_path

        image_list += [
            DatasetImage(
                relative_path=image_path,
                class_id=class_int,
                set_id=set_value,
                problem=problem_value,
            )
        ]

    return (image_list, class_dict)


# if the file is run directly, it will fully prepare the dataset from scratch
if __name__ == "__main__":

    # get the path
    if len(sys.argv) > 1:
        dataset_root_path = str(sys.argv[1])
    else:
        dataset_root_path = os.getcwd()

    # make sure path ends in a single trailing slash
    dataset_root_path = (dataset_root_path + "/").replace("//", "/")

    # start with the download
    download_dataset(dataset_root_path)

    if not verify_md5sums(dataset_root_path):
        print("md5sums do not match, exiting")
        sys.exit()

    extract_dataset(dataset_root_path)

    convert_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
