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


# It has train/val/test split, train/val data has label but test data has no label
# The labels are from 'scenes/CLEVR_train_scenes.json' and 'scenes/CLEVR_val_scenes.json', where they contain 'object' in the json file
# label = num_object - 3


#
# 1. constants
#

DATASET_URL_1 = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
DATASET_FILE_1 = "CLEVR_v1.0.zip"
DATASET_SUM_1 = "b11922020e72d0cd9154779b2d3d07d2"


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
    run_cmd_get_output(["unzip", "-qq", original_file_path])


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

    # class/label list (via json file)
    import json

    train_scenes_path = (
        dataset_root_path + "original_files/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
    )
    train_scenes_file = open(train_scenes_path)
    train_data = json.load(train_scenes_file)
    train_scenes = train_data["scenes"]

    val_scenes_path = (
        dataset_root_path + "original_files/CLEVR_v1.0/scenes/CLEVR_val_scenes.json"
    )
    val_scenes_file = open(val_scenes_path)
    val_data = json.load(val_scenes_file)
    val_scenes = val_data["scenes"]

    class_list = []
    for i in range(len(train_scenes)):
        class_list.append(len(train_scenes[i]["objects"]))
    class_set = set(class_list)

    # update class_dict
    for k, class_name in enumerate(sorted(class_set)):
        class_name = str(class_name)
        class_dict.update({class_name: k})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # train/test split
        if "train" in image_path:
            set_value = 0
        elif "val" in image_path:
            set_value = 1
        elif "test" in image_path:
            problem_value = 4

        if problem_value != 0:
            # if any problems, set class id to -1
            class_int = -1
            set_value = -1
        else:
            # get class name from image path and verify before setting class id
            class_from_path = image_path.split("/")[-1]
            class_name = class_from_path.split(".")[0]
            image_index = int(class_name.split("_")[-1])
            if "train" in class_name:
                assert train_scenes[image_index]["image_filename"] == class_from_path
                label_name = len(train_scenes[image_index]["objects"])
            elif "val" in class_name:
                assert val_scenes[image_index]["image_filename"] == class_from_path
                label_name = len(val_scenes[image_index]["objects"])

            label_name = str(label_name)
            assert label_name in class_dict
            class_int = class_dict[label_name]

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
