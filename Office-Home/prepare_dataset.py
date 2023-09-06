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
# 1) although the zip contains a csv and txt file, these are the same data in two different formats
#    the data is -just- the link to the original image (at the time)
# 2) there is no separate class name/id list or pairing with the images,
#    so all of the class names are in the folder structure


#
# 1. constants
#

DATASET_URL_1 = "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw"
DATASET_FILE_1 = "OfficeHomeDataset_10072016.zip"
DATASET_SUM_1 = "b1c14819770c4448fd5b6d931031c91c"

# real world contains a space in it in the zip, but the final name will have an underscore
SUBDATASETS = ["Art", "Clipart", "Product", "Real World"]

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
    remove_folder_or_file_and_check,
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

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again

    if not os.path.isfile(DATASET_FILE_1):
        run_cmd_get_output(["wget", DATASET_URL_1])


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    md5sum_result_1 = (
        DATASET_SUM_1
        == run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    )
    print(f"sum matches for {DATASET_FILE_1}: {md5sum_result_1}")

    # the return should only be True if all md5sums match
    return md5sum_result_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # clean out the original_files
    chdir_with_create(dataset_root_path)
    run_cmd_get_output(["rm", "-rf", "original_files"])

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["unzip", "-qq", original_file_path])

    print("create the subdatasets")
    for subdataset_name in SUBDATASETS:
        move_subdataset_folders(dataset_root_path, subdataset_name)


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    chdir_with_create(dataset_root_path)

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # the directory names are the class
    directories = get_unique_directories_from_files_list(image_path_list, 2)

    # since there is no given class id, use a sorted list
    class_dict: dict[str, int] = {}

    # class_names have underscores, so they will be left alone
    for i, class_name in enumerate(sorted(directories, key=str.casefold)):
        class_dict.update({class_name: i})

    image_list: list[DatasetImage] = []
    for image_path in image_path_list:
        problem_value = 0
        set_value = 0  # everything it is in

        class_id = None
        for class_name in class_dict.keys():
            if ("/" + class_name + "/") in image_path:
                class_id = class_dict[class_name]
                break  # exit after match
        assert not class_id is None

        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1
            set_value = -1

        image_list += [
            DatasetImage(
                relative_path=image_path,
                class_id=class_id,
                set_id=set_value,
                problem=problem_value,
            )
        ]

    return (image_list, class_dict)


def move_subdataset_folders(dataset_root_path: str, subdataset_name: str):
    """this code moves the subdataset folders into their appropriate subdataset (as a whole)"""

    print("  " + subdataset_name + "...")

    # make sure to replace the space with an underscore for the folder
    subdataset_path = (
        dataset_root_path + "../Office-Home_" + subdataset_name.replace(" ", "_") + "/"
    )

    # make sure the subdataset folder exists first
    chdir_with_create(subdataset_path)

    # first, clean out the subdataset directory
    remove_folder_or_file_and_check("database.sqlite")
    remove_folder_or_file_and_check("original_files")

    # then go into original_files
    chdir_with_create(subdataset_path + "original_files/")

    src_subdataset_folder = (
        dataset_root_path
        + "original_files/OfficeHomeDataset_10072016/"
        + subdataset_name
        + "/"
    )
    # destination should not have a trailing slash, to make sure folders get moved whole
    dst_subdataset_folder = subdataset_path + "original_files"

    run_cmd_get_output(["mv", src_subdataset_folder, dst_subdataset_folder])


def create_subset_databases(dataset_root_path: str):
    """instead of a single call for the super dataset,
    this function will call a similar function for each"""
    print("create_images_database for each subdataset")

    # loop through each subdataset
    for subdataset_name in SUBDATASETS:

        subdataset_path = (
            dataset_root_path
            + "../Office-Home_"
            + subdataset_name.replace(" ", "_")
            + "/"
        )
        print("*****  parse " + subdataset_name + " dataset  *****")

        # use the same call, but point to the subdataset_path
        create_images_database(
            subdataset_path,
            parse_dataset,
        )


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

    # create the sqlite file for each subdataset
    # NOTE: this calls the normal create_dataset function for each subdataset
    create_subset_databases(dataset_root_path)
