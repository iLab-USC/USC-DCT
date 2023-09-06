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
# 1) Multi-label facial attribute dataset, the specific task is: multi-class hair color
# 2) Classes taken are black hair, brown hair, blonde hair, gray hair, bald
# 3) Checked to make sure classes are mostly unbiased against gender
# 4) Only accepted samples that had only one of these attributes tagged as true

#
# 1. constants
#

# only 1 file is necessary for this example dataset
DATASET_URL_1 = "missing"
DATASET_FILE_1 = "img_align_celeba.zip"
DATASET_SUM_1 = "00d2c5bc6d35e252742224ab0c1e8fcb"

DATASET_FILE_2 = "list_attr_celeba.txt"
DATASET_SUM_2 = "1683cb81bb91ffcb6fa40ccc058891d1"

DATASET_FILE_3 = "identity_CelebA.txt"
DATASET_SUM_3 = "32bd1bd63d3c78cd57e08160ec5ed1e2"

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
    # chdir_with_create(dataset_root_path + "archives/")

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again
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
    md5sum_result_1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    md5sum_result_2 = run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]
    md5sum_result_3 = run_cmd_get_output(["md5sum", DATASET_FILE_3]).strip().split()[0]

    # the return should only be True if all md5sums match
    return (
        (md5sum_result_1 == DATASET_SUM_1)
        and (md5sum_result_2 == DATASET_SUM_2)
        and (md5sum_result_3 == DATASET_SUM_3)
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
    run_cmd_get_output(["unzip", "-qq", original_file_path])

    run_cmd_get_output(["cp", "../archives/" + DATASET_FILE_2, "."])
    run_cmd_get_output(["cp", "../archives/" + DATASET_FILE_3, "."])


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
    # get class names from directory structure
    path_to_attribute_annos = (
        dataset_root_path + "original_files/" + "list_attr_celeba.txt"
    )
    with open(path_to_attribute_annos, "r") as f:
        rows = f.readlines()
        num_images = int(rows[0].strip())
        header = rows[1]
        annos = rows[2:]

    assert len(annos) == num_images
    assert len(image_path_list) == num_images

    class_list = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
    class_anno_idx = {class_name: idx for idx, class_name in enumerate(header.split())}

    # update class_dict with class indices
    for k, class_name in enumerate(sorted(class_list)):
        class_dict.update({class_name: k})

    # get subject ids from file
    path_to_subjectid_annos = (
        dataset_root_path + "original_files/" + "identity_CelebA.txt"
    )
    with open(path_to_subjectid_annos, "r") as f:
        rows = f.readlines()
        image2subject = {
            "original_files/"
            + "img_align_celeba/"
            + r.split()[0].strip(): int(r.split()[1].strip())
            for r in rows
        }

    # get list of images for dataset
    used_images = {}
    for anno in annos:
        image_name = anno.split()[0].strip()
        all_annos = [int(val.strip()) for val in anno.split()[1:]]
        class_annos = [all_annos[class_anno_idx[c] - 1] for c in class_list]
        assert all([c == 1 or c == -1 for c in class_annos])
        if sum(class_annos) == -3:
            img_path = "original_files/" + "img_align_celeba/" + image_name
            assert img_path not in used_images
            class_name = [
                c for idx, c in enumerate(class_list) if class_annos[idx] == 1
            ]
            assert len(class_name) == 1
            used_images.update({img_path: class_name[0]})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value = 0, 0

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        # remove images in the non-classification/raw dataset
        if image_path not in used_images:
            problem_value = 4

        if problem_value != 0:
            # if any problems, set class id to -1
            class_id = -1
            set_value = -1
        else:
            # get class name from image path and verify before setting class id
            class_from_dict = used_images[image_path]
            assert class_from_dict in class_dict
            class_id = class_dict[class_from_dict]

        assert image_path in image2subject
        subject_id = image2subject[image_path]

        image_list += [
            DatasetImage(
                relative_path=image_path,
                class_id=class_id,
                set_id=set_value,
                problem=problem_value,
                subject_id=subject_id,
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
