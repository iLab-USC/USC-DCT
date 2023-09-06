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
# 1 no train/test split, so put all images as train image
# 2 no provided label_name to label index mapping, so create one use alphabeta sequence.
# 3 svg images were converted to png (one image was eliminated because conversion failed)
# 4 this code does not copy filelist/filelist.txt to converted_files/ but they also don't get used


#
# 1. constants
#

DATASET_URL_1 = (
    "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip"
)
DATASET_FILE_1 = "sketches_svg.zip"
DATASET_SUM_1 = "aa7d8ae9c8bf5f5cb6d28cee9741737c"


#
# 1. imports
#

import json  # for json file read
import os
import sys

import cairosvg
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

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again
    #
    # if not os.path.isfile(DATASET_FILE_1):
    #    run_cmd_get_output(["wget", DATASET_URL_1])


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
    run_cmd_get_output(["unzip", original_file_path])


def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """

    # change dir
    chdir_with_create(dataset_root_path + "converted_files/")

    # change .svg as .png
    original_file_path = dataset_root_path + "original_files/svg"
    for root, dirs, files in os.walk(original_file_path):
        for dir in dirs:
            os.makedirs(
                os.path.join(root.replace("original_files", "converted_files"), dir),
                exist_ok=True,
            )
        for file in files:
            if ".svg" in file:  # only process .svg files
                svg_file_name = os.path.join(root, file)
                png_file_name = svg_file_name.replace(
                    "original_files", "converted_files"
                ).replace(".svg", ".png")
                classname = svg_file_name.split("/")[-2]

                # set the background color, None=(transparent areas will become black)
                bg_color = "white"

                try:
                    cairosvg.svg2png(
                        url=svg_file_name,
                        write_to=png_file_name,
                        background_color=bg_color,
                    )
                except:
                    print("error converting file:", svg_file_name)


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    image_list: list[DatasetImage] = []
    class_dict: dict[str, int] = {}

    files_list = get_all_files_in_directory(
        dataset_root_path + "converted_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # create class list
    class_list = os.listdir(dataset_root_path + "converted_files/" + "svg")

    # update class_dict with class indices
    for k, class_name in enumerate(sorted(class_list, key=str.casefold)):
        class_dict.update({class_name: k})

    for i, img_name in enumerate(image_path_list):  # for each image path
        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + img_name):
            # if image is invalid, then add it to the list
            problem_value = 1
        else:
            problem_value = 0

        if problem_value != 0:
            # if any problems, set class id to -1
            img_set_id = -1
            img_label = -1
        else:
            # find imagelabel based on class_dict
            problem_value = 0
            img_label = class_dict[img_name.split("/")[2]]
            img_set_id = 0  # all train images

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

    
    convert_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
