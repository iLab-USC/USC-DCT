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


# 1 no train test split, so all images are treated as train
# 3 only contain label name, no label index, so created the label dict with alphabetical sequence


#
# 1. constants
#

DATASET_URL_1 = "missing"
DATASET_FILE_1 = "event_dataset.rar"
DATASET_SUM_1 = "22a1d43f12eebfa809b5425b706addbf"


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

    # need to extract with the full path, and quietly
    run_cmd_get_output(["unrar", "x", "-idq", original_file_path])


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

    # get label dict
    label_names = []
    for i, img_name in enumerate(image_path_list):  # for each image path
        img_label_name = img_name.split("/")[-1].split("_")[2]
        if img_label_name not in label_names:
            label_names.append(img_label_name)
    # update class_dict with class indices
    for i, class_name in enumerate(sorted(label_names, key=str.casefold)):
        class_dict.update({class_name: i})

    for i, img_name in enumerate(image_path_list):  # for each image path
        # find the image index
        img_label = class_dict[img_name.split("/")[-1].split("_")[2]]
        img_set_id = 0
        problem_value = 0
        image_list += [
            DatasetImage(
                relative_path=img_name,
                class_id=img_label,
                set_id=img_set_id,
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
