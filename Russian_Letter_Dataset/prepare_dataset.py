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
# 1) There are 4 datasets of russian letters included. All 4 will be used, because they contain the same 33 Russian
#    letters (although the class_id for the zip_letters set needs to be adjusted).
#    for datasets: letters, letter2, letters3
#       the filename contains is: [class_id]_[incrementing].png
#    for datasets: zip_letters
#       the filename contains is [upper/lowercase]_[class_id]_[background]_[incrementing].png
# 2) letters3 is completely duplicate elsewhere


#
# 1. constants
#

DATASET_URL_1 = "https://www.kaggle.com/datasets/olgabelitskaya/classification-of-handwritten-letters"
DATASET_FILE_1 = "archive.zip"
DATASET_SUM_1 = "393bb5ba533a05238b09c888916550fd"


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


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    # verify md5sums for each downloaded file
    md5sum_result_1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]

    # the return should only be True if all md5sums match
    return md5sum_result_1 == DATASET_SUM_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # extract the entire archive as-is into ./original_files/
    original_location = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["unzip", "-qq", original_location])


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

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # although there is a csv file for each of the letters (1-3) datasets, the file name
    # contains all of the necessary information. Instead, the letters.csv will be used to get the
    # russian glyphs

    letters_df = pd.read_csv(dataset_root_path + "original_files/letters.csv")
    # only keep the letter and label
    letters_df = letters_df[["letter", "label"]]
    # make sure there are 33 unique values
    letters_df = (
        letters_df.drop_duplicates().reset_index(drop=True).sort_values("label")
    )
    assert len(letters_df) == 33

    for i in range(len(letters_df)):
        class_dict.update({letters_df.loc[i].letter: int(letters_df.loc[i].label)})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        # there is only 1 set, training
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # dataset letters3 is completely duplicated elsewhere
        if "/letters3/" in image_path:
            problem_value = 3

        if problem_value != 0:
            # if any problems, set class id to None
            class_id = -1
            set_value = -1
        else:
            filename = image_path.split("/")[-1]
            filename_part = filename.split("_")
            # if part of the original letters (1-3) sets, then i starts at 1
            if "original_files/letters" in image_path:
                # the class id comes from the first part of the filename
                class_id = int(
                    filename_part[0]
                )  # and the class id started at 1 (but the same order)

            else:
                # else, for zip letters the class id comes from the second part
                class_id = int(filename_part[1]) + 1

            assert 1 <= class_id <= 33

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
