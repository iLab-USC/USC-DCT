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


# pgm image
# only training

#
# 1. constants
#

DATASET_URL_1 = "http://eprints.lincoln.ac.uk/id/eprint/16081/1/face.tar.gz"
DATASET_FILE_1 = "face.tar.gz"
DATASET_SUM_1 = "11011ab5dded043f5e4331711c2407c8"


#
# 1. imports
#


import os
import sys

from dataset_utils import (
    VALID_IMAGE_EXTENSIONS,
    DatasetImage,
    chdir_with_create,
    create_images_database,
    filter_file_list,
    fix_directory_permissions,
    get_all_files_in_directory,
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

    md5sum_result = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    return md5sum_result == DATASET_SUM_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

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

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # remove none-image file
    (image_path_list, _) = filter_file_list(files_list)

    # class/label list (via folder name)
    class_list = os.listdir(dataset_root_path + "original_files/")
    for k, class_name in enumerate(sorted(class_list)):
        class_dict.update({class_name: k})

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
        else:
            # get class name from image path and verify before setting class id
            class_from_path = image_path.split("/")[-2]
            assert class_from_path in class_dict
            class_int = class_dict[class_from_path]

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