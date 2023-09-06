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


# 1) 4 md5sum to check
# 2) test set has no labels
# 3) label and classname information in .csv file and class_list.txt


#
# 1. constants
#

# only 1 file is necessary for this example dataset
DATASET_URL_1 = "https://food-x.s3.amazonaws.com/annot.tar"
DATASET_FILE_1 = "annot.tar"
DATASET_SUM_1 = "0c632c543ceed0e70f0eb2db58eda3ab"

DATASET_URL_1 = "https://food-x.s3.amazonaws.com/test.tar"
DATASET_FILE_2 = "test.tar"
DATASET_SUM_2 = "32479146dd081d38895e46bb93fed58f"

DATASET_URL_3 = "https://food-x.s3.amazonaws.com/train.tar"
DATASET_FILE_3 = "train.tar"
DATASET_SUM_3 = "8e56440e365ee852dcb0953a9307e27f"

DATASET_URL_4 = "https://food-x.s3.amazonaws.com/val.tar"
DATASET_FILE_4 = "val.tar"
DATASET_SUM_4 = "fa9a4c1eb929835a0fe68734f4868d3b"

#
# 1. imports
#


import os
import re
import sys

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
    md5sum_result_2 = run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]
    md5sum_result_3 = run_cmd_get_output(["md5sum", DATASET_FILE_3]).strip().split()[0]
    md5sum_result_4 = run_cmd_get_output(["md5sum", DATASET_FILE_4]).strip().split()[0]

    # the return should only be True if all md5sums match
    return (
        md5sum_result_1 == DATASET_SUM_1
        and md5sum_result_2 == DATASET_SUM_2
        and md5sum_result_3 == DATASET_SUM_3
        and md5sum_result_4 == DATASET_SUM_4
    )


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # extract the entire archive as-is into ./original_files/
    for DATASET_FILE in [
        DATASET_FILE_1,
        DATASET_FILE_2,
        DATASET_FILE_3,
        DATASET_FILE_4,
    ]:
        original_file_path = dataset_root_path + "archives/" + DATASET_FILE
        run_cmd_get_output(["tar", "-xf", original_file_path])


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # generate the class_dict from class_list.txt
    with open(dataset_root_path + "original_files/class_list.txt") as ifile:
        class_list = ifile.readlines()
        class_dict = {
            line.strip("\n").split(" ")[1]: int(line.strip("\n").split(" ")[0])
            for line in class_list
        }

    # label:class mapping
    label_class_dict = dict()
    for k, v in class_dict.items():
        label_class_dict.update({v: k})

    train_label_dict = {}
    val_label_dict = {}
    with open(dataset_root_path + "original_files/train_info.csv", "r") as ifile:
        lines = ifile.readlines()
        train_label_dict = {
            line.split(",")[0]: int(line.split(",")[1].strip("\n")) for line in lines
        }
    with open(dataset_root_path + "original_files/val_info.csv", "r") as ifile:
        lines = ifile.readlines()
        val_label_dict = {
            line.split(",")[0]: int(line.split(",")[1].strip("\n")) for line in lines
        }

    img_label_dict = {**train_label_dict, **val_label_dict}

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if "val_set" in image_path:
            set_value = 1
        elif "test_set" in image_path:
            problem_value = 4

        if problem_value != 0:
            # if any problems, set class id to -1
            class_int = -1
            set_value = -1
        else:
            class_from_path = image_path.split("/")[-1]
            # print(class_from_path)
            assert class_from_path in img_label_dict
            class_int = img_label_dict[class_from_path]

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

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
