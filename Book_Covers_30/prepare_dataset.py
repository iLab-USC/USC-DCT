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
# 1) there are two files for each of the test/train splits. However, the listing file is the full file,
#    and includes the class names and ids.


#
# 1. constants
#

DATASET_URL_1 = "https://github.com/uchidalab/book-dataset"
DATASET_FILES = {
    "title30cat.zip": "436c33a71a65b5a4bfe71e9df0b56d7f",
    "bookcover30-labels-test.txt": "647a86a22f73189bc4c661649a5d18d6",
    "bookcover30-labels-train.txt": "6cf6cb08bde42683307c4358a318d1e0",
    "book30-listing-test.csv": "b1689a3b920023364f0489dff9b60995",
    "book30-listing-train.csv": "220d523df772d35589499e8338339391",
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
    for file in DATASET_FILES.keys():
        # verify md5sums for each downloaded file
        md5sum_result = run_cmd_get_output(["md5sum", file]).strip().split()[0]

        if md5sum_result != DATASET_FILES[file]:
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

    # extract the entire archive as-is into ./original_files/

    for file in DATASET_FILES.keys():
        original_file_path = dataset_root_path + "archives/" + file

        if file[-3:] == "zip":
            run_cmd_get_output(["unzip", "-qq", original_file_path])
        else:
            run_cmd_get_output(["cp", original_file_path, "."])


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    # encoding is not utf-8, so using latin1 instead
    # there is no header
    # columns are "[AMAZON INDEX (ASIN)}","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]"
    train_df = pd.read_csv(
        dataset_root_path + "original_files/book30-listing-train.csv",
        header=None,
        encoding="latin1",
        names=["index", "filename", "url", "title", "author", "class_id", "class_name"],
    )
    test_df = pd.read_csv(
        dataset_root_path + "original_files/book30-listing-test.csv",
        header=None,
        encoding="latin1",
        names=["index", "filename", "url", "title", "author", "class_id", "class_name"],
    )

    # make sure the class ids are names are the same
    assert set(list(train_df.class_id.unique())) == set(list(test_df.class_id.unique()))
    assert set(list(train_df.class_name.unique())) == set(
        list(test_df.class_name.unique())
    )

    # make sure there is no overlap of filenames
    test_filenames = list(test_df.filename)
    assert len(train_df[train_df.filename.isin(test_filenames)]) == 0

    # now, to speed up processing, turn the dataframes into dicts
    train_dict = {}
    for row in train_df.itertuples(index=False):
        train_dict.update({row.filename: int(row.class_id)})
    test_dict = {}
    for row in test_df.itertuples(index=False):
        test_dict.update({row.filename: int(row.class_id)})

    for class_id in sorted(list(train_df.class_id.unique())):
        class_name = list(
            train_df.loc[train_df.class_id == class_id].class_name.unique()
        )
        assert len(class_name) == 1
        class_name = class_name[0]
        class_dict.update({class_name: int(class_id)})

    # now get all of the files
    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        # there is only 1 set, training
        problem_value, set_value = 0, 0

        filename = image_path.split("/")[-1]

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if problem_value != 0:
            # if any problems, set class id to None
            class_id = -1
            set_value = -1
        else:

            if filename in train_dict:
                class_id = train_dict[filename]
            else:
                assert filename in test_dict
                class_id = test_dict[filename]
                set_value = 2

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
