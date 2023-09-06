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
# no train/val/test split, only training dataset.
# duplicate images in ./original_files/repository folder
# class_dict is derived from readme.md


#
# 1. constants
#

DATASET_URL_1 = "https://www.kaggle.com/datasets/jr2ngb/cataractdataset/download?datasetVersionNumber=2"
DATASET_FILE_1 = "archive.zip"
DATASET_SUM_1 = "2b8a95b5a4d9f9b332204e48ccf333f8"


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
    get_all_files_in_directory,
    get_unique_directories_from_files_list,
    is_image_valid,
    run_cmd_get_output,
)

#
# 2. functions with fixed APIs
#


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

    # extract the entire archive as-is into ./original_files/

    original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["unzip", "-qq", original_file_path])


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/
    """

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # remove none-image file
    (image_path_list, _) = filter_file_list(files_list)

    # class/label list (via file name)
    class_list = os.listdir(dataset_root_path + "original_files/" + "dataset")

    # update class_dict
    for k, class_name in enumerate(sorted(class_list)):
        class_name_list = class_name.split("_")
        class_name = " ".join(class_name_list[1:])
        class_dict.update({class_name: k + 1})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # Markinf duplicated dataset files as redundant
        if "repository" in image_path:
            problem_value = 3

        if problem_value != 0:
            # if any problems, set class id to -1
            class_id = -1
            set_value = -1
        else:
            # get class name from image path and verify before setting class id
            class_from_path = image_path.split("/")[-2]
            class_from_path_list = class_from_path.split("_")
            class_from_path = " ".join(class_from_path_list[1:])
            assert class_from_path in class_dict
            class_id = class_dict[class_from_path]

        image_list += [
            DatasetImage(
                relative_path=image_path,
                class_id=class_id,
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

    if not verify_md5sums(dataset_root_path):
        print("md5sums do not match, exiting")
        sys.exit()

    extract_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
