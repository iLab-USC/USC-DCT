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
# 1) This file actually creates sub-datasets for each of the "groups" that iNaturalist provides,
#    and inside those folders will be the images in original_files and the sqlite file
# 2) There are annotation json files that contain multiple lists which can be internally
#    validated (and are). However, all of the information necessary to get the class id and
#    name are in the folder structure itself
# 3) For Animalia, which is made from the species not selected by any of the other subdatasets,
#    the class ids are not contiguous.


#
# 1. constants
#

# train images
DATASET_URL_1 = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"
)
DATASET_FILE_1 = "train.tar.gz"
DATASET_SUM_1 = "e0526d53c7f7b2e3167b2b43bb2690ed"

# train annotations
DATASET_URL_2 = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz"
)
DATASET_FILE_2 = "train.json.tar.gz"
DATASET_SUM_2 = "38a7bb733f7a09214d44293460ec0021"

# val images
DATASET_URL_3 = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
DATASET_FILE_3 = "val.tar.gz"
DATASET_SUM_3 = "f6f6e0e242e3d4c9569ba56400938afc"

# val annotations
DATASET_URL_4 = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"
)
DATASET_FILE_4 = "val.json.tar.gz"
DATASET_SUM_4 = "4d761e0f6a86cc63e8f7afc91f6a8f0b"

# these are in alphabetical order, except for Animalia
SUBDATASETS = [
    "Actinopterygii",
    "Amphibia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia",
    "Animalia",  # last, because it uses the remaining files
]

#
# 1. imports
#

import json
import os
import sys
import time

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

    # don't download file again
    if not os.path.isfile(DATASET_FILE_1):
        run_cmd_get_output(["wget", DATASET_URL_1])

    if not os.path.isfile(DATASET_FILE_2):
        run_cmd_get_output(["wget", DATASET_URL_2])

    if not os.path.isfile(DATASET_FILE_3):
        run_cmd_get_output(["wget", DATASET_URL_3])

    if not os.path.isfile(DATASET_FILE_4):
        run_cmd_get_output(["wget", DATASET_URL_4])


def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums (will take a while)")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    md5sum_result_1 = (
        DATASET_SUM_1
        == run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]
    )
    print(f"sum matches for {DATASET_FILE_1}: {md5sum_result_1}")

    md5sum_result_2 = (
        DATASET_SUM_2
        == run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]
    )
    print(f"sum matches for {DATASET_FILE_2}: {md5sum_result_2}")

    md5sum_result_3 = (
        DATASET_SUM_3
        == run_cmd_get_output(["md5sum", DATASET_FILE_3]).strip().split()[0]
    )
    print(f"sum matches for {DATASET_FILE_3}: {md5sum_result_3}")

    md5sum_result_4 = (
        DATASET_SUM_4
        == run_cmd_get_output(["md5sum", DATASET_FILE_4]).strip().split()[0]
    )
    print(f"sum matches for {DATASET_FILE_4}: {md5sum_result_4}")

    return md5sum_result_1 and md5sum_result_2 and md5sum_result_3 and md5sum_result_4


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # fortunately, none of the extractions overlap each other,
    # so all will be extracted directly into original_files

    # first, extract the two json files
    print("extract the train/val json (annotation) files")
    original_location = dataset_root_path + "archives/" + DATASET_FILE_2
    run_cmd_get_output(["tar", "-xf", original_location])
    original_location = dataset_root_path + "archives/" + DATASET_FILE_4
    run_cmd_get_output(["tar", "-xf", original_location])

    print("extract train images (longest wait)")
    original_training = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["tar", "-xf", original_training])

    print("extract val images (less long wait)")
    original_validation = dataset_root_path + "archives/" + DATASET_FILE_3
    run_cmd_get_output(["tar", "-xf", original_validation])


def prepare_subdatasets(dataset_root_path: str):
    """prepare all of the subdatasets"""

    print("prepare the subdatasets")

    # get a list of all of the directories that were extracted (these will be moved)
    train_directories = os.listdir(dataset_root_path + "original_files/train/")
    val_directories = os.listdir(dataset_root_path + "original_files/val/")

    # verify that all of the same classes exist, and that there are 10000
    all_match_1 = list(set(train_directories) - set(val_directories))
    all_match_2 = list(set(val_directories) - set(train_directories))
    if (len(all_match_1) > 0) or (len(all_match_2) > 0):
        print("error extracting, as the sub folders don't match")
        sys.exit()

    if len(train_directories) != 10000:
        print("error in expected count of folders")
        sys.exit()

    # new variable to modify (by removing the datasets pulled from)
    all_directories = train_directories

    # now, move the files to their proper subdatasets
    for subdataset_name in SUBDATASETS:
        all_directories = move_subdataset_folders(
            dataset_root_path, subdataset_name, all_directories
        )

    # verify that no folders remain
    train_directories = os.listdir(dataset_root_path + "original_files/train/")
    val_directories = os.listdir(dataset_root_path + "original_files/val/")
    if (len(train_directories) > 0) or (len(val_directories) > 0):
        print("not all folders were moved into place")
        sys.exit()

    # at this point only the train.json, val.json and the empty train and val folders remain


def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    chdir_with_create(dataset_root_path)

    # make sure sqlite is erased
    remove_folder_or_file_and_check("database.sqlite")

    files_list = get_all_files_in_directory(
        dataset_root_path + "original_files/", dataset_root_path
    )

    # splitting the file list into images and non-images
    (image_path_list, _) = filter_file_list(files_list)

    # the directory names (class) are the same for both train and val
    directories = get_unique_directories_from_files_list(image_path_list, 2)

    # all directories have the (class_id at the beginning, but the items need to be sorted)
    directories = sorted(directories)

    # build the class_dict
    # store a separate tuple for matching the directory
    classes_tuple = []
    class_dict: dict[str, int] = {}
    for directory in directories:
        # the structure of all of the directories are the same
        class_id = int(directory[:5])
        class_name = directory[6:].replace("_", " ")
        classes_tuple += [(directory, class_id)]
        class_dict.update({class_name: class_id})

    # build the images_list
    image_list: list[DatasetImage] = []
    for image_path in image_path_list:
        problem_value = 0

        class_id = None
        for class_tuple_i in range(len(classes_tuple)):
            # make sure the image_path contains the desired folder (regardless of train/val)
            if classes_tuple[class_tuple_i][0] in image_path:
                class_id = classes_tuple[class_tuple_i][1]
                break  # break when a match is found
        assert not class_id is None

        if "train/" in image_path:
            set_value = 0  # train
        else:
            set_value = 1  # val

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


def create_subset_databases(dataset_root_path: str):
    """instead of a single call for the super dataset,
    this function will call a similar function for each"""
    print("create_images_database for each subdataset (will take a really long time)")

    # loop through each subdataset
    for subdataset_name in SUBDATASETS:

        subdataset_path = dataset_root_path + "../iNaturalist_" + subdataset_name + "/"
        print("*****  parse " + subdataset_name + " dataset  *****")

        # use the same call, but point to the subdataset_path
        create_images_database(
            subdataset_path,
            parse_dataset,
        )


def verify_all_images_using_json_files(dataset_root_path: str):
    """verify that all of the images exist within the json files, and that the json files are consistent"""
    chdir_with_create(dataset_root_path + "original_files/")

    print("verify the internal consistency of train.json and val.json")
    file_contents = []
    with open("train.json") as file_h:
        file_contents = file_h.readlines()
    train_json = json.loads(file_contents[0])
    if not check_json_consistency(train_json):
        print("issue with internal consistency of train.json")
        sys.exit()

    with open("val.json") as file_h:
        file_contents = file_h.readlines()
    val_json = json.loads(file_contents[0])
    if not check_json_consistency(val_json):
        print("issue with internal consistency of val.json")
        sys.exit()

    # now the json files are verified to be internally consistent
    # however, it will now make sure that each image provided actually exists inside
    # the train.json and val.json (using the entire path)
    # for each subdataset individually (since the files have already been moved)

    # extract just the file_name from the images list
    train_images = []
    for i in range(len(train_json["images"])):
        train_images += [train_json["images"][i]["file_name"]]
    val_images = []
    for i in range(len(val_json["images"])):
        val_images += [val_json["images"][i]["file_name"]]

    print("verify files for subdatasets")
    for subdataset_name in SUBDATASETS:
        # verify the files, and return the lists of what remains
        (train_images, val_images) = verify_subdataset_files(
            dataset_root_path, subdataset_name, train_images, val_images
        )

    # make sure no images are left unaccounted for
    assert (len(train_images) == 0) and (len(val_images) == 0)


def verify_subdataset_files(
    dataset_root_path: str, subdataset_name: str, train_images: list, val_images: list
) -> tuple[list, list]:
    """This verifies all of the files for each subdataset"""
    print("  " + subdataset_name + "...")

    subdataset_path = dataset_root_path + "../iNaturalist_" + subdataset_name + "/"
    subdataset_files = get_all_files_in_directory(subdataset_path + "original_files/")
    (subdataset_images, subdataset_non_images) = filter_file_list(subdataset_files)

    # there should be no non-images
    assert len(subdataset_non_images) == 0

    # strip out the root_path
    for i in range(len(subdataset_images)):
        loc_i = subdataset_images[i].find("/original_files/")
        assert loc_i >= 0
        # keep only the path after original_files
        subdataset_images[i] = subdataset_images[i][(loc_i + 16) :]

    left_in_train = list(set(train_images) - set(subdataset_images))
    left_in_val = list(set(val_images) - set(subdataset_images))
    # remove both train and val from subdataset
    left_in_sub = list((set(subdataset_images) - set(train_images)) - set(val_images))

    # there should be no images unaccounted for in the subdataset
    assert len(left_in_sub) == 0

    # return the lists of remaining files
    return (left_in_train, left_in_val)


def move_subdataset_folders(
    dataset_root_path: str, subdataset_name: str, all_directories: list[str]
) -> list[str]:
    """This moves all of the appropriate folders from the iNaturalist/original_files/
    to the iNaturalist_[Name]/original_files/"""

    print("  " + subdataset_name + "...")
    subdataset_path = dataset_root_path + "../iNaturalist_" + subdataset_name + "/"

    # make sure the subdataset folder exists first
    chdir_with_create(subdataset_path)

    assert os.path.isdir(subdataset_path)

    # first, clean out the subdataset directory
    remove_folder_or_file_and_check("database.sqlite")
    remove_folder_or_file_and_check("original_files")

    # then go into original_files
    chdir_with_create(subdataset_path + "original_files/")

    subdataset_directories = []
    for i in reversed(range(len(all_directories))):
        if subdataset_name in all_directories[i]:
            subdataset_directories += [all_directories[i]]
            # remove from the main list (to make it easier to do Animalia at the end)
            del all_directories[i]

    print("    classes:", len(subdataset_directories))

    # destination folders (no trailing slash, so that each folder gets moved as a whole)
    dst_train_folder = subdataset_path + "original_files/train"
    dst_val_folder = subdataset_path + "original_files/val"

    # make sure both destinations exist
    chdir_with_create(dst_train_folder)
    chdir_with_create(dst_val_folder)

    # move each species_directory
    for species_directory in subdataset_directories:
        src_train_folder = (
            dataset_root_path + "original_files/train/" + species_directory
        )
        src_val_folder = dataset_root_path + "original_files/val/" + species_directory

        # make sure source folders exist before moving
        assert os.path.isdir(src_train_folder)
        assert os.path.isdir(src_val_folder)

        # move the train and val folders to the subdataset
        run_cmd_get_output(["mv", src_train_folder, dst_train_folder])
        run_cmd_get_output(["mv", src_val_folder, dst_val_folder])

    # return the remaining list
    return all_directories


def check_json_consistency(annotations_json: dict) -> bool:
    """since the folder structure contains both the full label and the class_id (category_id),
    it will instead make sure that everything is internally consistent with the annotation files provided"""

    # it is much faster to operating on a dataframe than loop through everything
    # so, the data will be loaded this way
    # there are 3 lists that provide the annotation data: ['images', 'annotations', 'categories']
    # and all 3 lists will be used to cross check the results against each other
    df_images = pd.DataFrame(data=annotations_json["images"])
    df_annotations = pd.DataFrame(data=annotations_json["annotations"])
    df_categories = pd.DataFrame(data=annotations_json["categories"])

    # verify the images ids all exist in the annotations list by storing the category_id given by df_annotations for that image_id
    df_images["category_id"] = df_images.id.map(
        df_annotations.set_index("image_id")["category_id"]
    )

    # get the image_dir_name from df_categories by mapping the category_id
    df_images["image_dir_name"] = df_images.category_id.map(
        df_categories.set_index("id")["image_dir_name"]
    )

    # check if the image_dir_name is contained within the file_name for each image
    df_images["matches_image_dir_name"] = df_images.apply(
        lambda row: row["image_dir_name"] in row["file_name"], axis=1
    )

    # finally, check all of the results of the test
    results = df_images["matches_image_dir_name"].unique()

    # return the check of if there is only one unique value, and the value is True
    return (len(results) == 1) and (results[0])


# if the file is run directly, it will fully prepare the dataset from scratch
if __name__ == "__main__":

    # get the path
    if len(sys.argv) > 1:
        dataset_root_path = str(sys.argv[1])
    else:
        dataset_root_path = os.getcwd()

    # make sure path ends in a single trailing slash
    dataset_root_path = (dataset_root_path + "/").replace("//", "/")

    print("sit back and relax, this will take about 3.5 hours")

    # start with the download
    download_dataset(dataset_root_path)

    if not verify_md5sums(dataset_root_path):
        print("md5sums do not match, exiting")
        sys.exit()

    extract_dataset(dataset_root_path)

    # this prepares the subdataset folders
    prepare_subdatasets(dataset_root_path)

    # this is a superset, and the subsets will be done differently
    verify_all_images_using_json_files(dataset_root_path)

    # create the sqlite file for each subdataset
    # NOTE: this calls the normal create_dataset function for each subdataset
    create_subset_databases(dataset_root_path)
