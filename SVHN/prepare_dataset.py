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
# 1) Images are stored as numpy arrays in 3 matlab files, with one of them being the test set.
# 2) The mat files designate class_id 10 for digit 0. The same, strange, class_id will be used.


#
# 1. constants
#

DATASET_URL_1 = "http://ufldl.stanford.edu/housenumbers/"
DATASET_FILES = {
    "extra_32x32.mat": "a93ce644f1a588dc4d68dda5feec44a7",
    "test_32x32.mat": "eb5a983be6a315427106f1b164d9cef3",
    "train_32x32.mat": "e26dedcc434d2e4c54c9b2d4a06d8373",
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
from PIL import Image
from scipy.io import loadmat

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

    all_hashes_match = True
    for file in DATASET_FILES.keys():
        md5sum_result = run_cmd_get_output(["md5sum", file]).strip().split()[0]
        if md5sum_result != DATASET_FILES[file]:
            all_hashes_match = False

    return all_hashes_match


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("copy dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    for file in DATASET_FILES.keys():
        # extract the entire archive as-is into ./original_files/
        original_location = dataset_root_path + "archives/" + file
        run_cmd_get_output(["cp", original_location, "."])


def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """
    print("convert dataset")

    # change dir
    chdir_with_create(dataset_root_path + "converted_files/")

    for file in DATASET_FILES.keys():
        matfile = loadmat(dataset_root_path + "original_files/" + file)

        # data is stored in 'X' [32x32x3xcount] and 'y' [count, 1]
        folder_name = file.split("_")[0]

        chdir_with_create(dataset_root_path + "converted_files/" + folder_name + "/")

        all_image_data = matfile["X"][:]

        for i in range(matfile["y"].shape[0]):
            # set the filename as: [class_id]_[image_i].png
            class_id = matfile["y"][i][0]

            filename = str(class_id) + "_" + str(i) + ".png"

            image = Image.fromarray(all_image_data[:, :, :, i])
            image.save(filename)


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    # although the class id is 10 for digit 0, the name is obviously still "0"
    for i in range(1, 11):
        class_name = str(i) if i < 10 else "0"
        class_dict.update({class_name: i})

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )
    files_list2 = get_all_files_in_directory(
        dataset_root_path + "converted_files/", dataset_root_path
    )
    files_list = files_list + files_list2

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        # there is only 1 set, training
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if "/test/" in image_path:
            set_value = 2

        if problem_value != 0:
            # if any problems, set class id to None
            class_id = -1
            set_value = -1
        else:
            filename = image_path.split("/")[-1]
            filename_part = filename.split("_")
            class_id = int(filename_part[0])

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
