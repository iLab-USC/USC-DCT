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
# 1 all image in jpg folder, imagelabels.mat contains labelsl setid.mat contains tran(to read .mat, use scipy)
# import scipy.io
# mat = scipy.io.loadmat('./imagelabels.mat')
# 2 the official website do not provide label name and label index mapping, there is one in a github link, use this one.
# https://github.com/MAbdelhamid2001/Image-Classifier/blob/main/label_map.json
# 3 the imagelabel start from 0, while image name start from 1, there is one miss match.


#
# 1. constants
#

DATASET_URL_1 = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
DATASET_FILE_1 = "fgvc-aircraft-2013b.tar.gz"
DATASET_SUM_1 = "d4acdd33327262359767eeaa97a4f732"

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

    # load train image information
    with open(
        dataset_root_path
        + "original_files/fgvc-aircraft-2013b/data/images_family_train.txt"
    ) as f:
        train_info_list = f.readlines()
    train_info_dict = {}
    for ele in train_info_list:
        image_id = ele.split(" ")[0]
        train_info_dict[image_id] = ele.split(image_id + " ")[-1].split("\n")[0]

    # load val information
    with open(
        dataset_root_path
        + "original_files/fgvc-aircraft-2013b/data/images_family_val.txt"
    ) as f:
        val_info_list = f.readlines()
    val_info_dict = {}
    for ele in val_info_list:
        image_id = ele.split(" ")[0]
        val_info_dict[image_id] = ele.split(image_id + " ")[-1].split("\n")[0]

    # load test image information
    with open(
        dataset_root_path
        + "original_files/fgvc-aircraft-2013b/data/images_family_test.txt"
    ) as f:
        test_info_list = f.readlines()
    test_info_dict = {}
    for ele in test_info_list:
        image_id = ele.split(" ")[0]
        test_info_dict[image_id] = ele.split(image_id + " ")[-1].split("\n")[0]

    # load label labe dict
    with open(
        dataset_root_path + "original_files/fgvc-aircraft-2013b/data/families.txt"
    ) as f:
        family_load = f.readlines()
    family_load = [ele.split("\n")[0] for ele in family_load]

    for class_id, class_name in enumerate(sorted(family_load, key=str.casefold)):
        class_dict.update({class_name: int(class_id)})

    for i, img_name in enumerate(image_path_list):  # for each image path
        # find the image index
        image_id = img_name.split("/")[-1].split(".")[0]

        # find split dataset infobased on index
        if image_id in train_info_dict.keys():
            img_set_id = 0
            img_label = class_dict[train_info_dict[image_id]]
        elif image_id in val_info_dict.keys():
            img_set_id = 1
            img_label = class_dict[val_info_dict[image_id]]
        elif image_id in test_info_dict.keys():
            img_set_id = 2
            img_label = class_dict[test_info_dict[image_id]]
        else:
            print("error, image do not belons to train/val/test")

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
