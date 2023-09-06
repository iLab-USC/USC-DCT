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
# 1) per documentation, class_ids are 0-42
# 2) no class name for the signs seems to be apparent in the files or website


#
# 1. constants
#

DATASET_URL_1 = "https://benchmark.ini.rub.de/gtsrb_dataset.html"
DATASET_FILE_SUM_DICT = {
    "GTSRB_Final_Test_GT.zip": "fe31e9c9270bbcd7b84b7f21a9d9d9e5",
    "GTSRB_Final_Test_Images.zip": "c7e4e6327067d32654124b0fe9e82185",
    "GTSRB_Final_Training_Images.zip": "f33fd80ac59bff73c82d25ab499e03a3",
}


#
# 1. imports
#


import os
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
    chdir_with_create(dataset_root_path + "archives/")


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    all_sums_match = True
    for file in DATASET_FILE_SUM_DICT.keys():
        # verify md5sums for each downloaded file
        md5sum_result_1 = run_cmd_get_output(["md5sum", file]).strip().split()[0]
        if md5sum_result_1 != DATASET_FILE_SUM_DICT[file]:
            all_sums_match = False

    # the return should only be True if all md5sums match
    return all_sums_match


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    for file in DATASET_FILE_SUM_DICT.keys():
        original_file_path = dataset_root_path + "archives/" + file
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

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # no class names given, just ids
    for i in range(43):
        class_dict.update({str(i): i})

    test_labels_df = pd.read_csv(
        dataset_root_path + "original_files/GT-final_test.csv", delimiter=";"
    )

    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if problem_value != 0:
            # if any problems, set class id to None
            class_id = -1
            set_value = -1
        else:
            if "Final_Test" in image_path:
                set_value = 2
                filename = image_path.split("/")[4]
                values = list(
                    test_labels_df[test_labels_df.Filename == filename].ClassId.values
                )
                # make sure only one match was made
                assert len(values) == 1
                class_id = int(values[0])
            else:
                class_id = int(image_path.split("/")[4])

            assert 0 <= class_id <= 42

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
