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
# 1 this is the understanding of cars-annos.mat
# [('relative_im_path', 'O'), ('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('test', 'O')]
# 2 they provide a class list in .mat format, in class label of the image, they use 0~196, then assume the class list is from 0 to 196 mapping to class id.


#
# 1. constants
#

DATASET_URL_1 = "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz"
DATASET_FILE_1 = "car_ims.tgz"
DATASET_SUM_1 = "d5c8f0aa497503f355e17dc7886c3f14"

DATASET_URL_2 = "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"
DATASET_FILE_2 = "cars_test_annos_withlabels.mat"
DATASET_SUM_2 = "b0a2b23655a3edd16d84508592a98d10"

DATASET_URL_3 = ""
DATASET_FILE_3 = "cars_annos.mat"
DATASET_SUM_3 = "b407c6086d669747186bd1d764ff9dbc"


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
    md5sum_result2 = run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]
    md5sum_result3 = run_cmd_get_output(["md5sum", DATASET_FILE_3]).strip().split()[0]

    # the return should only be True if all md5sums match
    return (
        md5sum_result1 == DATASET_SUM_1
        and md5sum_result2 == DATASET_SUM_2
        and md5sum_result3 == DATASET_SUM_3
    )


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
    run_cmd_get_output(
        ["cp", "../archives/cars_test_annos_withlabels.mat", "."]
    )  # copy setid to split train/test/val
    run_cmd_get_output(
        ["cp", "../archives/cars_annos.mat", "."]
    )  # copy setid to split train/test/val


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
    # image_path_list = sorted(image_path_list) # start from
    # load the image label
    imageinfos_mat = scipy.io.loadmat(
        dataset_root_path + "original_files/cars_annos.mat"
    )
    imageinfos = imageinfos_mat["annotations"][0]

    for imageinfo in imageinfos:
        img_name = "original_files/" + imageinfo[0][0]
        img_label = int(imageinfo[-2])
        img_set_id = 0 if int(imageinfo[-1]) == 0 else 2  # only train or test
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
    imagelabels = imageinfos_mat["class_names"][0]
    for i, label in enumerate(imagelabels):
        class_dict.update({label[0]: i + 1})

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
