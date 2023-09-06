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

# The original dataset does not contain the images, but instead has the URLs for the images.
# This is a problem, as those URLs can eventually go dark.
# Therefore, the convert_dataset will try to download all of the files that it can, and store
# notes about what it could not download (or if the downloaded data was not an image).
# However, if the archive file Yoga-82-images.zip exists, then it is extracted and used.
# There is a train/test split.


#
# 1. constants
#

DATASET_URL_1 = "missing"
DATASET_FILE_1 = "Yoga-82.rar"
DATASET_SUM_1 = "d4f5b438a40a73535a2334f352b90eb7"

# this file was created by ilab
DATASET_FILE_2 = "Yoga-82-images.zip"
DATASET_SUM_2 = "7f28aff2042ee42cf8435cc361a31a4d"


#
# 1. imports
#

import concurrent.futures
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pandas as pd
import requests
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
    md5sum_result_1 = run_cmd_get_output(["md5sum", DATASET_FILE_1]).strip().split()[0]

    # only check the second file if it exists
    md5sum_result_2 = True
    if os.path.isfile(DATASET_FILE_2):
        md5sum_result_2 = (
            DATASET_SUM_2
            == run_cmd_get_output(["md5sum", DATASET_FILE_2]).strip().split()[0]
        )

    # the return should only be True if all md5sums match
    return (md5sum_result_1 == DATASET_SUM_1) and md5sum_result_2


def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    # extract the entire archive as-is into ./original_files/
    original_location = dataset_root_path + "archives/" + DATASET_FILE_1
    run_cmd_get_output(["unrar", "x", "-idq", original_location])

    # extract the entire archive as-is into ./original_files/
    original_location = dataset_root_path + "archives/" + DATASET_FILE_2
    if os.path.isfile(original_location):
        run_cmd_get_output(["unzip", "-qq", original_location])


def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """

    if os.path.isdir(dataset_root_path + "original_files/images/"):
        print("images folder already exists, no conversion (download) necessary")
        return

    print("conversion will involve getting all of the original files from the urls")

    txt_files = os.listdir(
        dataset_root_path + "original_files/Yoga-82/yoga_dataset_links/"
    )

    chdir_with_create(dataset_root_path + "original_files/images/")

    yoga_df = pd.DataFrame(columns=["folder", "file_name", "url", "status_code"])

    print("extract the folder/file/url lists...")
    for i, txt_file in enumerate(sorted(txt_files, key=str.casefold)):

        # skip the non txt files
        if txt_file.split(".")[1] != "txt":
            continue

        txt_file_lines = []
        with open(
            dataset_root_path + "original_files/Yoga-82/yoga_dataset_links/" + txt_file
        ) as file_h:
            txt_file_lines = file_h.readlines()

        for i, line in enumerate(txt_file_lines):
            # this will work for the earlier sets, but not for all
            # line_data = line.split()
            line_data = []

            # instead find the http and split from there
            split_i = line.find("http")
            line_data += [line[:split_i].strip()]
            line_data += [line[(split_i - 1) :].strip()]

            save_folder = line_data[0].split("/")[0].strip()
            save_filename = line_data[0].split("/")[1].strip()
            url = line_data[1].strip()

            yoga_df.loc[len(yoga_df)] = [  # type: ignore
                save_folder,
                save_filename,
                url,
                0,  # status code 0 for now
            ]

    def get_image(i):
        assert 0 <= i < len(yoga_df)
        entry = yoga_df.loc[i]

        save_folder: str = (
            dataset_root_path
            + "original_files/images/"
            + entry.folder  # type: ignore
            + "/"
        )

        # make sure the folder exists
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        save_filename: str = save_folder + entry.file_name  # type: ignore

        if os.path.isfile(save_filename):
            if is_image_valid(save_filename):
                # if the file already exists, then the status code was 200
                yoga_df.loc[i, "status_code"] = 200
                return
            else:
                # if the file wasn't valid, then remove to try again
                os.remove(save_filename)

        # try to get the data, otherwise skip
        try:
            response = requests.get(entry.url, timeout=3)  # type: ignore
            yoga_df.loc[i, "status_code"] = response.status_code
            try:
                image = Image.open(BytesIO(response.content))

                # all files in the provided list have a jpg extension, but the source file
                # could be something else, so convert to RGB
                if image.mode in ["RGBA", "P", "LA"]:
                    image = image.convert("RGB")

                image.save(save_filename)
            except Exception as e:
                # failed while creating image
                yoga_df.loc[i, "status_code"] = str(e)  # -2
        except Exception as e:
            # failed while trying to get image
            yoga_df.loc[i, "status_code"] = str(e)  # -1

    print("start downloading images...")
    with ThreadPoolExecutor(max_workers=50) as e:
        shuffled_list = yoga_df.index.values.tolist()
        random.shuffle(shuffled_list)

        for i in shuffled_list:
            e.submit(get_image, i)

    print("done downloading")

    # save results to an excel file (in case there are any delimiter issues)
    chdir_with_create(dataset_root_path + "original_files/images/")
    yoga_df.to_excel("all_images.xlsx", index=False)


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

    train_df = pd.read_csv(
        dataset_root_path + "original_files/Yoga-82/yoga_train.txt",
        names=["combined", "label_of_6", "label_of_20", "label_of_82"],
    )
    test_df = pd.read_csv(
        dataset_root_path + "original_files/Yoga-82/yoga_test.txt",
        names=["combined", "label_of_6", "label_of_20", "label_of_82"],
    )

    # verify the label_ids match
    assert set(list(train_df.label_of_82.unique())) == set(
        list(test_df.label_of_82.unique())
    )

    train_df["class_name"] = train_df.combined.apply(lambda x: x.split("/")[0].strip())

    train_images = list(train_df.combined)
    test_images = list(test_df.combined)

    for class_id in sorted(list(train_df.label_of_82.unique())):
        class_name = list(
            train_df[train_df.label_of_82 == class_id].class_name.unique()
        )
        assert len(class_name) == 1
        class_name = class_name[0]
        class_dict.update({class_name: int(class_id)})

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
            class_from_path = image_path.split("/")[2]
            assert class_from_path in class_dict
            class_id = class_dict[class_from_path]

            class_plus_image_name = class_from_path + "/" + image_path.split("/")[3]
            if class_plus_image_name in train_images:
                assert not class_plus_image_name in test_images
            else:
                assert not class_plus_image_name in train_images
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

    convert_dataset(dataset_root_path)

    # create the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
