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
# 1) Class names and indices were hard coded from the readme.txt the dataset provides

#
# 1. constants
#

DATASET_FILE_1 = "rvl-cdip.tar.gz"
DATASET_SUM_1 = "d641dd4866145316a1ed628b420d8b6c"
DATASET_LABELS = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific report",
    6: "scientific publication",
    7: "specification",
    8: "file folder",
    9: "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}

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
    md5sum_result = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]

    # the return should only be True if all md5sums match
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

    # get list of train/val/test images:labels
    relative_prefix = "original_files/" + "images/"
    train_txt = dataset_root_path + "original_files/" + "labels/" + "train.txt"
    with open(train_txt, "r") as fr:
        rows = fr.readlines()
        train_files = {relative_prefix + r.split()[0]: int(r.split()[1]) for r in rows}

    val_txt = dataset_root_path + "original_files/" + "labels/" + "val.txt"
    with open(val_txt, "r") as fr:
        rows = fr.readlines()
        val_files = {relative_prefix + r.split()[0]: int(r.split()[1]) for r in rows}

    test_txt = dataset_root_path + "original_files/" + "labels/" + "test.txt"
    with open(test_txt, "r") as fr:
        rows = fr.readlines()
        test_files = {relative_prefix + r.split()[0]: int(r.split()[1]) for r in rows}

    # update class_dict with class indices
    for k, class_name in DATASET_LABELS.items():
        class_dict.update({class_name: k})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # check which set the image is in, get its class index, set value
        image_in_dataset = 0
        if image_path in train_files:
            image_in_dataset += 1
            class_int = train_files[image_path]
            set_value = 0
        if image_path in val_files:
            image_in_dataset += 1
            class_int = val_files[image_path]
            set_value = 1
        if image_path in test_files:
            image_in_dataset += 1
            class_int = test_files[image_path]
            set_value = 2

        # make sure an image is not in multiple sets
        if image_in_dataset > 0:
            assert image_in_dataset == 1
            assert class_int in DATASET_LABELS
        else:
            problem_value = 4  # set problem 4 if no label foudn in text files

        if problem_value != 0:
            # if any problems, set class id to -1
            class_int = -1
            set_value = -1

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
