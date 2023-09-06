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


# 1 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/action_guidelines/index.html guidelines
# 2 problem=7 for multiple labels. So if two or more of the boolean flags are set, then the image should be problem=7
# 3 install xmltodict for xml read
# 4 use alphabeta sequence create the class name to class index mapping


#
# 1. constants
#


DATASET_URL_1 = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)
DATASET_FILE_1 = "VOCtrainval_11-May-2012.tar"
DATASET_SUM_1 = "6cd6e144f989b92b3379bac3b3de84fd"


#
# 1. imports
#


import os
import sys

import xmltodict  # version 0.13.0 needed
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

    md5sum_result1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]

    return md5sum_result1 == DATASET_SUM_1


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1

    run_cmd_get_output(["tar", "-xf", original_file_path])


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

    # load train/val image split
    with open(
        dataset_root_path
        + "original_files/VOCdevkit/VOC2012/ImageSets/Action/train.txt"
    ) as f:
        train_set_raw = f.readlines()
    train_set_list = [ele.split("\n")[0] for ele in train_set_raw]
    with open(
        dataset_root_path + "original_files/VOCdevkit/VOC2012/ImageSets/Action/val.txt"
    ) as f:
        val_set_raw = f.readlines()
    val_set_list = [ele.split("\n")[0] for ele in val_set_raw]

    # create class list
    sample_xml = os.path.join(
        dataset_root_path,
        "original_files/VOCdevkit/VOC2012/",
        "Annotations",
        train_set_list[0] + ".xml",
    )
    with open(sample_xml) as file:
        file_data = file.read()
        dict_data = xmltodict.parse(file_data)

    class_list = list(dict_data["annotation"]["object"]["actions"].keys())
    # update class_dict with class indices
    for k, class_name in enumerate(sorted(class_list)):
        class_dict.update({class_name: k})

    for i, img_name in enumerate(image_path_list):  # for each image path
        # check if the image is valid, if not then the class is None
        file_name = img_name.split("/")[-1].split(".")[0]
        problem_value = 0
        img_set_id = 0

        if not is_image_valid(dataset_root_path + img_name):
            # if image is invalid, then add it to the list
            problem_value = 1
        elif (
            img_name.split("/")[3] == "SegmentationObject"
            or img_name.split("/")[3] == "SegmentationClass"
        ):  # not relevant
            problem_value = 6
        elif (
            file_name not in train_set_list and file_name not in val_set_list
        ):  # not relevant
            problem_value = 6
        else:  # useful image, no trivial issue, check if multiple labels
            xml_path = os.path.join(
                dataset_root_path,
                "original_files/VOCdevkit/VOC2012/",
                "Annotations",
                file_name + ".xml",
            )

            with open(xml_path) as file:
                file_data = file.read()
                dict_data = xmltodict.parse(file_data)

            # single person
            if type(dict_data["annotation"]["object"]) is dict:
                assert dict_data["annotation"]["object"]["name"] == "person"
                assert "actions" in dict_data["annotation"]["object"]
                action_dict = dict_data["annotation"]["object"]["actions"]

                if (
                    sum([int(ele) for ele in list(action_dict.values())]) > 1
                ):  # multilabel
                    problem_value = 7
                else:
                    problem_value = 0
                    img_label = class_dict[
                        list(action_dict.keys())[list(action_dict.values()).index("1")]
                    ]

                    if file_name in train_set_list:
                        img_set_id = 0
                    elif file_name in val_set_list:
                        img_set_id = 1
                    else:
                        print("error image not in train/val" + file_name)

            elif type(dict_data["annotation"]["object"]) is list:
                # multiple person, see if they are consistent

                img_label_log = []
                action_dict = {}

                # go over each individual person
                for each_person in dict_data["annotation"]["object"]:
                    assert each_person["name"] == "person"
                    assert "actions" in each_person
                    action_dict = each_person["actions"]

                    # multilabel
                    if sum([int(ele) for ele in list(action_dict.values())]) > 1:
                        problem_value = 7
                        break  # break the for loop
                    else:
                        img_label_log.append(
                            class_dict[
                                list(action_dict.keys())[
                                    list(action_dict.values()).index("1")
                                ]
                            ]
                        )

                # all human actions are equal
                if problem_value != 7 and img_label_log.count(img_label_log[0]) == len(
                    img_label_log
                ):
                    problem_value = 0
                    img_label = class_dict[
                        list(action_dict.keys())[list(action_dict.values()).index("1")]
                    ]

                    if file_name in train_set_list:
                        img_set_id = 0
                    elif file_name in val_set_list:
                        img_set_id = 1
                    else:
                        print("error image not in train/val" + file_name)
                else:
                    problem_value = 7

        if problem_value != 0:
            # if any problems, set class id to -1
            img_set_id = -1
            img_label = -1

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

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
