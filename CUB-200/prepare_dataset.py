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


#
# 1


#
# 1. constants
#


DATASET_URL_1 = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
DATASET_FILE_1 = "CUB_200_2011.tgz"
DATASET_SUM_1 = "97eceeb196236b17998738112f37df78"

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
    fix_directory_permissions,
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

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again
    #
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
    run_cmd_get_output(["tar", "-xf", original_file_path])

    fix_directory_permissions(dataset_root_path + "original_files/")


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []

    class_dict: dict[str, int] = {}

    # load image information
    with open(dataset_root_path + "original_files/CUB_200_2011/images.txt") as f:
        files_list = f.readlines()

    # load image label
    with open(
        dataset_root_path + "original_files/CUB_200_2011/image_class_labels.txt"
    ) as f:
        image_class_labels = f.readlines()
    image_id_2_class_label_dict = {}
    for label_content in image_class_labels:
        image_id_2_class_label_dict[int(label_content.split(" ")[0])] = int(
            label_content.split(" ")[-1].split("\n")[0]
        )

    # load train/test split label
    with open(
        dataset_root_path + "original_files/CUB_200_2011/train_test_split.txt"
    ) as f:
        train_test_split = f.readlines()
    image_id_2_split_dict = {}
    for (
        split_content
    ) in (
        train_test_split
    ):  # mapping original 1(training) as 0, original 0(testing) as 2
        train_or_test = int(
            split_content.split(" ")[-1].split("\n")[0]
        )  # 1: training, 0, test
        image_id_2_split_dict[int(split_content.split(" ")[0])] = (
            0 if train_or_test == 1 else 2
        )

    for file_content in files_list:
        image_id = int(file_content.split(" ")[0])  # e.g., 1
        img_name = (
            "original_files/CUB_200_2011/images/"
            + file_content.split(" ")[-1].split("\n")[0]
        )  # e.g., 'original_files/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
        img_label = image_id_2_class_label_dict[image_id]
        img_set_id = image_id_2_split_dict[image_id]
        problem_value = 0
        image_list += [
            DatasetImage(
                relative_path=img_name,
                class_id=img_label,
                set_id=img_set_id,
                problem=problem_value,
            )
        ]

    # load label_map
    with open(dataset_root_path + "original_files/CUB_200_2011/classes.txt") as f:
        label_map = f.readlines()

    # update class_dict with class indices
    for class_name in label_map:
        class_dict.update(
            {class_name.split(" ")[-1].split("\n")[0]: int(class_name.split(" ")[0])}
        )

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
