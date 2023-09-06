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


#
# 1. constants
#

DATASET_URL_1 = (
    "https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB"
)
DATASET_FILE_AND_MD5 = [
    ("camelyonpatch_level_2_split_test_meta.csv", "3455fd69135b66734e1008f3af684566"),
    ("camelyonpatch_level_2_split_test_x.h5.gz", "d8c2d60d490dbd479f8199bdfa0cf6ec"),
    ("camelyonpatch_level_2_split_test_y.h5.gz", "60a7035772fbdb7f34eb86d4420cf66a"),
    ("camelyonpatch_level_2_split_train_meta.csv", "5a3dd671e465cfd74b5b822125e65b0a"),
    ("camelyonpatch_level_2_split_train_x.h5.gz", "1571f514728f59376b705fc836ff4b63"),
    ("camelyonpatch_level_2_split_train_y.h5.gz", "35c2d7259d906cfc8143347bb8e05be7"),
    ("camelyonpatch_level_2_split_valid_meta.csv", "67589e00a4a37ec317f2d1932c7502ca"),
    ("camelyonpatch_level_2_split_valid_x.h5.gz", "d5b63470df7cfa627aeec8b9dc0c066e"),
    ("camelyonpatch_level_2_split_valid_y.h5.gz", "2b85f58b927af9964a4c15b8f7e8f179"),
]

#
# 1. imports
#


import os
import sys

import h5py
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
from PIL import Image

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

    # the return should only be True if all md5sums match

    # verify md5sums for each downloaded file
    result = True
    for pair in DATASET_FILE_AND_MD5:
        md5sum_result = run_cmd_get_output(["md5sum", pair[0]]).strip().split()[0]
        if pair[1] != md5sum_result:
            result = False
            print(pair[0])

    # the return should only be True if all md5sums match
    return result


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    for i in [1, 2, 4, 5, 7, 8]:
        original_file_path = DATASET_FILE_AND_MD5[i][0]
        run_cmd_get_output(["cp", "-p", "../archives/" + original_file_path, "./"])
        run_cmd_get_output(["gunzip", "-d", original_file_path])


def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """
    print("convert dataset")

    # change dir
    chdir_with_create(dataset_root_path + "converted_files/")

    # create a dir for each class
    chdir_with_create(dataset_root_path + "converted_files/0/")
    chdir_with_create(dataset_root_path + "converted_files/1/")

    # move back to load h5 files
    chdir_with_create(dataset_root_path + "original_files/")

    for i in [1, 4, 7]:
        original_file_path = DATASET_FILE_AND_MD5[i][0][:-7]
        file_x = h5py.File(original_file_path + "x.h5", "r")
        file_y = h5py.File(original_file_path + "y.h5", "r")
        dset = file_x["x"]
        label = file_y["y"]

        for j, im in enumerate(dset):  # type: ignore
            class_id = int(label[j][0][0][0])  # type: ignore

            image_path = dataset_root_path + f"converted_files/{class_id}/{i}_{j}.png"
            image = Image.fromarray(im)
            image.save(image_path)

        file_x.close()
        file_y.close()


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    chdir_with_create(dataset_root_path + "original_files/")

    class_dict = {"negative": 0, "positive": 1}

    files_list = get_all_files_in_directory(
        dataset_root_path + "converted_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # go through each image, match to flag problems,
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
            # get class name from image path and verify before setting class id
            class_id = int(image_path.split("/")[1])

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

    
    convert_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
