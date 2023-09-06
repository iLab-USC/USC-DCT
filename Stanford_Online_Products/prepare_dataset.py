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


# 1 Used the 12 superclass name as the class.
# 2 will take a long time because the number of images is 120,053


#
# 1. constants
#

DATASET_URL_1 = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
DATASET_FILE_1 = "Stanford_Online_Products.zip"
DATASET_SUM_1 = "7f73d41a2f44250d4779881525aea32e"


#
# 1. imports
#

import json  # for json file read
import os
import sys

import scipy.io  # added for .mat load
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
    chdir_with_create(dataset_root_path + "archives/")


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    # verify md5sums for each downloaded file

    md5sum_result1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    # the return should only be True if all md5sums match
    return md5sum_result1 == DATASET_SUM_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # extract the entire archive as-is into ./original_files/
    original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1

    run_cmd_get_output(["unzip", "-qq", original_file_path])


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    # load train information
    with open(
        dataset_root_path + "original_files/Stanford_Online_Products/Ebay_train.txt"
    ) as f:
        train_info_list = f.readlines()
    for ele in train_info_list[1:]:
        img_set_id = 0  # for train
        super_class_id = ele.split(" ")[2]
        img_name = (
            "original_files/Stanford_Online_Products/"
            + ele.split(" ")[3].split("\n")[0]
        )
        problem_value = 0
        image_list += [
            DatasetImage(
                relative_path=img_name,
                class_id=int(super_class_id),
                set_id=img_set_id,
                problem=problem_value,
            )
        ]
    # load test information
    with open(
        dataset_root_path + "original_files/Stanford_Online_Products/Ebay_test.txt"
    ) as f:
        test_info_list = f.readlines()
    for ele in test_info_list[1:]:
        img_set_id = 2  # for train
        super_class_id = ele.split(" ")[2]
        img_name = (
            "original_files/Stanford_Online_Products/"
            + ele.split(" ")[3].split("\n")[0]
        )
        problem_value = 0
        image_list += [
            DatasetImage(
                relative_path=img_name,
                class_id=int(super_class_id),
                set_id=img_set_id,
                problem=problem_value,
            )
        ]

    # get label dict
    for ele in train_info_list[1:]:
        super_class_id = ele.split(" ")[2]
        class_name = ele.split(" ")[3].split("\n")[0].split("_")[0]
        if class_name not in class_dict.keys():
            class_dict.update({class_name: int(super_class_id)})

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
