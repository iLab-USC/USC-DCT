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
# --Dataset has 16 different "labels" for images in contrast to the 11 different text/source defined labels
#   $ Ignoring all image labels (unless noted otherwise) and using concatenation of all text sources
#
# --Text Labels split into multiple different "quality" files (e.g. 1_Good, 2_Good, 1_Okay, etc.)
#   $ Ignoring separations any concatenating all with the same base label together (as mentioned in source)
#
# -- Text files with label "query" are used to reflect the visual aid presented online, and do not provide labels
#   $ Directly ignoring all "Query_files"


#
# 1. constants
#

DATASET_URL_1 = (
    "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
)
DATASET_FILE_1 = "oxbuild_images-v1.tgz"
DATASET_SUM_1 = "8f0017cc5d671f35011d6968269fbf47"


DATASET_URL_2 = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz"
DATASET_FILE_2 = "gt_files_170407.tgz"
DATASET_SUM_2 = "06c4dee0ad954f6ee122c82390548537"

#
# 1. imports
#


import os
import re
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

    # verify md5sums for each downloaded file
    md5sum_result_1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    md5sum_result_2 = run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]

    # the return should only be True if all md5sums match
    return (md5sum_result_1 == DATASET_SUM_1) and (md5sum_result_2 == DATASET_SUM_2)


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # extract the entire archive as-is into ./original_files/

    original_file1_path = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["tar", "-xf", original_file1_path])

    original_file2_path = dataset_root_path + "archives/" + DATASET_FILE_2
    run_cmd_get_output(["tar", "-xf", original_file2_path])


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
    (image_path_list, text_path_list) = filter_file_list(files_list)

    # generate list of classes AND list of file labels
    class_list = []
    text2img_dict: dict[str, int] = {}
    for filePath in text_path_list:
        # remove all filepaths, break at counting number (pattern: .../.../LABEL_#_...)
        className = re.split(r"\d+", filePath.replace("original_files/", ""))[0][:-1]
        # Collect Unique Class Names
        if className not in class_list:
            class_list.append(className)

    # creating class_dict, and text2img_dict
    for id, classStr in enumerate(sorted(class_list, key=str.casefold)):
        class_dict.update({classStr: id})

        # Create list of files denoted for the given classStr
        textLabels = []
        for fileText in sorted(text_path_list):
            # If class is in file name (and not a "Query" file), add to collection, else ignore
            if classStr in fileText and "query" not in fileText:
                for x in (
                    open(str(dataset_root_path + fileText), "r").read().split("\n")[:-1]
                ):
                    textLabels.append(x)
            else:
                pass
        # Generate dict with the image name: id#
        n = 0
        for txtImg in textLabels:
            n += 1
            text2img_dict.update({"original_files/" + txtImg + ".jpg": id})

    # go through each image, ensure valid, flag any problems, else find label
    for image_path in sorted(image_path_list):
        problem_value, set_value = 0, 0

        if not is_image_valid(dataset_root_path + image_path):
            problem_value = 1
            class_int = -1

        # If not in listed named files
        elif image_path not in text2img_dict.keys():
            class_int = -1
            problem_value = 4
        else:
            class_int = text2img_dict[image_path]

        if problem_value != 0:
            set_value = -1

        assert isinstance(class_int, int)

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
