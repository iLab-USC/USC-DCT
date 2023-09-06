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
# 1) Did not extract test.zip.* files since kaggle website suggests no labels exists for those
# 2) Image names have subject id, so be careful when splitting train/val/test sets.
# 3) Dataset labels are given in the kaggle website and hard coded as a constant, therefore not sorted alphabetically

#
# 1. constants
#

DATASET_URL_1 = "https://kaggle.com/competitions/diabetic-retinopathy-detection"
DATASET_FILE_1 = "diabetic-retinopathy-detection.zip"
DATASET_SUM_1 = "596cf4ecabf92e5e621ac7e9f9181471"

DATASET_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

#
# 1. imports
#


import os
import subprocess
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
    original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1

    # unzip the outer file
    print("unzip the outer file (will take a while)")
    run_cmd_get_output(["unzip", "-qq", original_file_path])

    # combine zip parts into one zip file
    filenames = sorted(
        [
            f
            for f in os.listdir(dataset_root_path + "original_files/")
            if "train.zip." in f
        ]
    )

    print("combine the split files for training data")
    combined_zip_path = dataset_root_path + "original_files/" + "combined_train.zip"
    with open(combined_zip_path, "wb") as outfile:
        for fname in filenames:
            with open(dataset_root_path + "original_files/" + fname, "rb") as infile:
                ## Read data in smaller chunks in case machine has low memory
                while True:
                    data_to_copy = infile.read(16 * 4096)
                    if not data_to_copy:
                        break
                    outfile.write(data_to_copy)

    # unzip combined zip file
    print("unzip the combined training zip file")
    run_cmd_get_output(["unzip", "-qq", combined_zip_path])

    # unzip the trainLabels.csv
    run_cmd_get_output(
        ["unzip", "-qq", dataset_root_path + "original_files/trainLabels.csv.zip"]
    )


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

    # read labels
    with open(dataset_root_path + "original_files/trainLabels.csv", "r") as fr:
        rows = fr.readlines()
        anno_dict = {
            "original_files/train/"
            + r.strip().split(",")[0]
            + ".jpeg": int(r.strip().split(",")[1])
            for r in rows[1:]
        }

    # update class_dict with class indices
    for idx, class_name in DATASET_LABELS.items():
        class_dict.update({class_name: idx})

    # go through each image, match to flag problems,
    for image_path in image_path_list:
        problem_value, set_value, subject_id = 0, 0, -1

        # check if the image is valid, if not then the class is None
        if not is_image_valid(dataset_root_path + image_path):
            # if image is invalid, then add it to the list
            problem_value = 1

        if problem_value != 0:
            # if any problems, set class id to None
            class_int = -1
            set_value = -1
        else:
            # get class name from image path and verify before setting class id
            class_int = anno_dict[image_path]
            assert class_int in set(class_dict.values())

        try:
            subject_id = int(image_path.split("/")[-1].split("_")[0])
        except:
            print("Subject id cannot be parsed: " + image_path)

        image_list += [
            DatasetImage(
                relative_path=image_path,
                class_id=class_int,
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

    if not verify_md5sums(dataset_root_path):
        print("md5sums do not match, exiting")
        sys.exit()

    extract_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
