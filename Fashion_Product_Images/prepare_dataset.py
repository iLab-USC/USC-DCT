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


# 1) under the dir/Fashion_Product_Images_Dataset/original_files/fashion-dataset/,
# there is a duplicate dir fashion-dataset
# 2) information about the labels(class_names) contained in the styles.csv and use sub_category as
# the class_name as before


#
# 1. constants
#

# only 1 file is necessary for this example dataset
DATASET_URL_1 = "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/download?datasetVersionNumber=1"
DATASET_FILE_1 = "archive.zip"
DATASET_SUM_1 = "1e146312734ba7bf88195bd00ed8bc64"


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

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # load the styles.csv file to get the class labels and ids
    styles_df = pd.read_csv(
        dataset_root_path + "original_files/" + "fashion-dataset/" + "styles.csv",
        sep=",",
        usecols=["id", "subCategory"],
    )
    class_list = pd.unique(styles_df.iloc[:, 1])

    # update class_dict with class indices
    for k, class_name in enumerate(sorted(class_list)):
        class_dict.update({class_name: k})

    # create the img_id to class_id mapping
    label_dict = styles_df.set_index("id").to_dict(orient="index")
    label_dict = {k: v["subCategory"] for k, v in label_dict.items()}

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # Mark inf duplicated dataset files as redundant
        if "fashion-dataset/fashion-dataset/" in image_path:
            # print(image_path)
            problem_value = 3

        # train, val, test split

        if problem_value != 0:
            # if any problems, set class id to -1
            class_int = -1
            set_value = -1
        else:
            # get class name from image path and verify before setting class id
            img_from_path = image_path.split("/")[-1]
            # strip the file descriptor (.png or .jpeg) from the path
            img_from_path = int(img_from_path[: img_from_path.find(".")])
            if img_from_path in label_dict:
                class_int = class_dict[label_dict[img_from_path]]
            else:
                class_int = -1
                problem_value = 4

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

    if not verify_md5sums(dataset_root_path):
        print("md5sums do not match, exiting")
        sys.exit()

    extract_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
