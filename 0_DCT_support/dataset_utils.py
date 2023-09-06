"""
This file will contain all of the helper functionality that is the same regardless of dataset.

This file should NOT be modified.
"""

import hashlib
import multiprocessing
import os
import random
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from glob import glob
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from PIL import Image

# pillow does not seem to have an enum for valid extensions
# so they are collected here (svg was removed, since pillow cannot open it)
VALID_IMAGE_EXTENSIONS = [
    "bmp",
    "gif",
    "jfif",
    "jpeg",
    "jpg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "webp",
]
VALID_SET_ID_LIST = [-1, 0, 1, 2]
VALID_PROBLEM_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]

NUM_WORKERS = 30

MIN_PIXELS = 16  # for problem 8

IMAGES_DATABASE_FILE = "database.sqlite"

if sys.version_info < (3, 9):
    raise SystemExit("Only Python 3.9 and above is supported")


@dataclass()
class DatasetImage(object):
    # data required for every image
    relative_path: str
    class_id: int
    set_id: int  # named to prevent issues in sqlite calls
    problem: int

    # since subject_id is rare, default to not requiring it
    subject_id: int = -1

    # data that is calculated later
    file_size: int = -1
    file_hash: str = ""
    image_hash: str = ""
    image_width: int = -1
    image_height: int = -1
    image_mode: str = ""
    image_format: str = ""

    def validate(self, dataset_root_path) -> bool:
        """validate for consistency"""

        # first just check the types and data and if the file exists
        first_pass = self.validate_types()
        second_pass = self.validate_data()
        third_pass = self.does_file_exist(dataset_root_path)

        return first_pass and second_pass and third_pass

    def validate_types(self) -> bool:
        """validate the types are correct"""

        if not isinstance(self.relative_path, str):
            print(self, "relative_path is not str")
            return False

        if not isinstance(self.class_id, int):
            print(self, "class_id is not int")
            return False

        if not isinstance(self.set_id, int):
            print(self, "set_id is not int")
            return False

        if not isinstance(self.problem, int):
            print(self, "problem is not int")
            return False

        if not isinstance(self.subject_id, int):
            print(self, "subject_id is not int")
            return False

        return True

    def validate_data(self) -> bool:
        """validate all of the data"""
        if len(self.relative_path) == 0:
            print(self, "relative_path is empty")
            return False

        if self.class_id < -1:
            print(self, "class_id is < -1")
            return False

        if not self.set_id in VALID_SET_ID_LIST:
            print(self, "set_id is not in", VALID_SET_ID_LIST)
            return False

        if not get_file_extension(self.relative_path) in VALID_IMAGE_EXTENSIONS:
            print(self, "is a non image file (according to the extension)")
            return False

        if not self.problem in VALID_PROBLEM_LIST:
            print(self, "problem is not in", VALID_PROBLEM_LIST)
            return False

        # if there is a problem, the class_id must also indicate that
        if (self.problem > 0) and (self.class_id != -1):
            print(self, "problem and class_id mismatch")
            return False

        # if the class_id says problem, a problem must be given
        if (self.class_id == -1) and (self.problem == 0):
            print(self, "problem and class_id mismatch")
            return False

        # if a problem is given, then set_id should be -1
        if (self.problem > 0) and (self.set_id >= 0):
            print(self, "problem and set_id mismatch")
            return False

        # if the set_id=-1, then there should be a problem given
        if (self.set_id == -1) and (self.problem == 0):
            print(self, "problem and set_id mismatch")
            return False

        if self.subject_id < -1:
            print(self, "subject_id is < -1")
            return False

        return True

    def does_file_exist(self, dataset_root_path: str) -> bool:
        """check if file exists"""
        if not os.path.isfile(dataset_root_path + self.relative_path):
            print(self, "file does not exist")
            self.file_size = -1
            return False

        # go ahead and get the file_size
        self.file_size = os.path.getsize(dataset_root_path + self.relative_path)

        return True

    def get_file_data(self, dataset_root_path: str):
        """get the hashes and validate the image"""

        # if the file hasn't been tested, then exit
        if self.file_size < 0:
            return

        image_path = dataset_root_path + self.relative_path

        # as long as the file exists, there will be a file hash
        self.file_hash = get_hash_of_file(image_path)

        # do a quick check first on validity
        if is_image_valid(image_path):

            # if the image is valid, then get the remaining details
            try:
                with Image.open(image_path) as image_h:
                    # get the image_hash
                    self.image_hash = get_hash_of_image(image_h)

                    # get other details
                    self.image_width = image_h.width
                    self.image_height = image_h.height
                    self.image_mode = image_h.mode
                    self.image_format = (
                        image_h.format if (not image_h.format is None) else ""
                    )

                # if no current problem, and the image is too small, then
                # give it problem 8
                if (self.problem == 0) and (
                    (self.image_height < MIN_PIXELS) | (self.image_width < MIN_PIXELS)
                ):
                    self.problem = 8

                # exit, when successful
                return
            except:
                pass

        # these are set if the image is invalid either by the fast check,
        # or it fails while getting the hash
        self.problem = 1
        self.set_id = -1
        self.class_id = -1
        self.image_hash = ""
        self.image_width = -1
        self.image_height = -1
        self.image_mode = ""
        self.image_format = ""


def create_images_database(
    dataset_root_path: str,
    parse_dataset: Callable[[str], tuple[list[DatasetImage], dict[str, int]]],
):
    """This will create the database using the two list functions."""

    # remove the sqlite files, if they exist
    if os.path.isfile(dataset_root_path + IMAGES_DATABASE_FILE):
        if not remove_folder_or_file_and_check(
            dataset_root_path + IMAGES_DATABASE_FILE
        ):
            sys.exit()
    if os.path.isfile(dataset_root_path + "ilab.sqlite"):
        if not remove_folder_or_file_and_check(dataset_root_path + "ilab.sqlite"):
            sys.exit()

    check_if_all_archive_files_are_used(dataset_root_path)

    # get the lists
    (image_list, classes_dict) = parse_dataset(dataset_root_path)

    # do a first check on the results returned
    assert validate_list_and_dict(dataset_root_path, image_list, classes_dict)

    # get all remaining data for images
    image_list = get_remaining_image_details(dataset_root_path, image_list)

    # convert list to dataframe (makes it easier to save to sqlite)
    df_images = DataFrame(data=image_list)

    # make sure that at least for every duplicate hash, it has the same file_size
    assert len(df_images.file_hash.unique()) == len(
        df_images[["file_hash", "file_size"]].drop_duplicates()
    )

    # convert to list[(class_name, class_id)]
    classes_list = list(classes_dict.items())
    # make sure it is sorted by class_id
    classes_list: list[tuple[str, int]] = sorted(classes_list, key=lambda x: x[1])
    # order to match the standard way that a dict is turned into a list
    df_classes = DataFrame(data=classes_list, columns=["class_name", "class_id"])
    # reorder the columns to a more sensible order
    df_classes = df_classes[["class_id", "class_name"]]

    # calculate the statistics before proceeding
    assert calculate_statistics(dataset_root_path, df_images, df_classes)

    # finally, save the database
    print(f"create {IMAGES_DATABASE_FILE}")

    # save to sqlite file (replacing any tables in-place)
    with sqlite3.connect(dataset_root_path + IMAGES_DATABASE_FILE) as conn:
        df_images.to_sql("images", conn, if_exists="replace", index=False)
        df_classes.to_sql("classes", conn, if_exists="replace", index=False)
        conn.commit()

    print("run additional checks on hashes")
    find_folders_full_of_duplicates(df_images)
    assert are_duplicates_real(df_images)
    problem_zero_dupes(df_images)

    # create the examples folder
    create_examples_directory(dataset_root_path)

    # remove the cache again (to make directory cleaner when done)
    remove_folder_or_file_and_check("__pycache__")


def problem_zero_dupes(
    images: DataFrame, silent: bool = False
) -> tuple[list[str], int]:
    """get stats on problem=0 dupes"""

    dupe_hashes = images[images.problem == 0].file_hash.value_counts()
    # keep only those that are actual duplicates
    dupe_hashes = dupe_hashes[dupe_hashes > 1]

    if len(dupe_hashes) == 0:
        return ([], 0)

    if not silent:
        print("there are", len(dupe_hashes), "(problem=0) hashes that are not unique")

    # store only the hash
    dupe_hashes = list(dupe_hashes.index)

    # group by hash and class_id, and do the following:
    # 1) count the number of images with each combination: .size()
    # 2) if there is more than 0 images for a combination, set the value as a 1: .gt(0).mul(1)
    # 3) unstack the combos, and wherever there wasn't a pair, set the value as 0: .unstack(fill_value=0)
    results = (
        images[(images.file_hash.isin(dupe_hashes)) & (images.problem == 0)]
        .groupby(["file_hash", "class_id"])
        .size()
        .gt(0)
        .mul(1)
        .unstack(fill_value=0)
    )
    # 4) sum the results along the rows (this will count the number of hash/class_id pairs for each hash)
    results = results.sum(axis=1)
    # 5) keep only the hashes that have at least 2 class_id pairs
    results = results[results > 1]

    # store the results
    dupe_hashes_with_more_than_one_class_id = list(results.index)
    dupe_hashes_image_count = len(
        images[
            (images.file_hash.isin(dupe_hashes_with_more_than_one_class_id))
            & (images.problem == 0)
        ]
    )

    if (not silent) and (len(dupe_hashes_with_more_than_one_class_id) > 0):
        print("there are some file_hashes that appear in more than one class")
        print("hash count:", len(dupe_hashes_with_more_than_one_class_id))
        print("image count:", dupe_hashes_image_count)

    return (dupe_hashes_with_more_than_one_class_id, dupe_hashes_image_count)


def are_duplicates_real(images: DataFrame) -> bool:
    """check if duplicates are actual duplicates"""

    duplicate_hashes = list(images.loc[images.problem == 3].file_hash.values)
    if len(duplicate_hashes) == 0:
        # there are no duplicates marked
        return True

    print("verify duplicates (problem=3) are actual duplicates")

    non_duplicate_hashes = list(images.loc[(images.problem != 3)].file_hash.values)

    not_actual_duplicates = list(set(duplicate_hashes) - set(non_duplicate_hashes))
    if len(not_actual_duplicates) == 0:
        print("yes, all duplicates were found elsewhere (problem!=3)")
        return True

    print(
        "there are",
        len(not_actual_duplicates),
        "images marked as problem=3 that are not real duplicates",
    )

    list_of_not_actual_duplicate_paths = list(
        images.loc[images.file_hash.isin(not_actual_duplicates)].relative_path.values
    )

    count_to_show = min(20, len(not_actual_duplicates))
    print(f"here are the first {count_to_show}:")
    for i in range(count_to_show):
        print(" ", list_of_not_actual_duplicate_paths[i])

    return False


def get_details_for_one_image(image: DatasetImage) -> DatasetImage:
    """get the details for a single image"""
    # this has to live as a main function to be called by map_async

    # janky way to get dataset_root_path
    global global_root_path

    image.get_file_data(global_root_path)

    return image


def get_remaining_image_details(
    dataset_root_path: str, image_list: list[DatasetImage]
) -> list[DatasetImage]:
    """calculate the file hashes and get the image details"""

    print("get all remaining image details")
    t_start = time.perf_counter()

    # janky way to pass dataset_root_path
    global global_root_path
    global_root_path = dataset_root_path

    # how many workers to use in parallel (either 20 or 80% of the cpu count)
    workers = min(NUM_WORKERS, int(0.95 * multiprocessing.cpu_count()))

    print(f"get image details using {workers} workers...")
    with multiprocessing.Pool(workers) as pool:
        new_image_list = pool.map(get_details_for_one_image, image_list)

    assert len(new_image_list) == len(image_list)

    print("took", round(time.perf_counter() - t_start), "sec")

    return new_image_list


def get_count_of_all_images_in_folder(
    dataset_root_path: str, remove_examples: bool = True
) -> int:
    """this is for making sure all images have been caught"""

    # no reason to strip out the root_path, since it's just used for counting
    all_files = get_all_files_in_directory(dataset_root_path)
    (all_images, _) = filter_file_list(all_files)

    # remove the ones in 'examples/'
    if remove_examples:
        all_images = [x for x in all_images if (x.find("examples/") < 0)]

    return len(all_images)


def is_list_in_alphabetical_order(class_list: list[str]) -> bool:
    """tests if list is in alphabetical order"""

    # must specify key=str.casefold in order to ignore case
    sorted_class_names = sorted(class_list, key=str.casefold)

    class_names_in_alphabetical_order = True
    for i in range(len(class_list)):
        if sorted_class_names[i] != class_list[i]:
            class_names_in_alphabetical_order = False
            break

    return class_names_in_alphabetical_order


def calculate_statistics(
    dataset_root_path: str, images: DataFrame, classes: DataFrame
) -> bool:
    """this will get statistics on the dataset to give us information"""

    print("#" * 70)
    print("statistics:")
    print("  image statistics:")
    print("    images total:", len(images))

    # check independently
    found_images = get_count_of_all_images_in_folder(dataset_root_path)
    print("    images found independently:", found_images)
    original_image_count = get_count_of_all_images_in_folder(
        dataset_root_path + "original_files/", False
    )
    print("      in original_files:", original_image_count)
    if os.path.isdir(dataset_root_path + "converted_files/"):
        converted_image_count = get_count_of_all_images_in_folder(
            dataset_root_path + "converted_files/", False
        )
        print("      in converted_files:", converted_image_count)

    if len(images) != found_images:
        print("not all images are accounted for")
        return False
    print("    problem-free images:", len(images[images.problem == 0]))
    print("      images in training set  :", len(images[images.set_id == 0]))
    print("      images in validation set:", len(images[images.set_id == 1]))
    print("      images in testing set   :", len(images[images.set_id == 2]))
    print("    problematic images:", len(images[images.problem != 0]))

    # if there are some problems, then split out stats
    if len(images[images.problem != 0]) > 0:
        print("      problematic stats:")
        problem_types = sorted(list(images[images.problem != 0].problem.unique()))
        for problem_type in problem_types:
            print(
                f"        problem={problem_type} images:",
                len(images[images.problem == problem_type]),
            )
    print("      class_id=-1 images:", len(images[images.class_id == -1]))
    print("      set_id=-1 images:", len(images[images.set_id == -1]))
    print(
        "      all 3 (problem/class_id/set_id) images:",
        len(
            images[
                (images.set_id == -1) & (images.class_id == -1) & (images.problem != 0)
            ]
        ),
    )

    # now do the class statistics
    print("  class statistics:")

    unique_class_ids = sorted(list(classes.class_id.unique()))  # type: ignore
    unique_class_names = list(classes.class_name.unique())
    number_of_classes = len(unique_class_ids)
    print("    classes (>=0):", number_of_classes)

    class_names_in_class_id_order = []
    for class_id in unique_class_ids:
        # get the class name
        class_name = str(classes.loc[classes.class_id == class_id].class_name.values[0])

        # store the class name, in order of class_id
        class_names_in_class_id_order += [class_name]

    longest_class_name = max(unique_class_names, key=len)
    shortest_class_name = min(unique_class_names, key=len)
    longest_class_name_len = min(len(longest_class_name), 40)

    few_count = min(100, number_of_classes)
    few_classes_list = unique_class_ids[:few_count]
    print(f"    first {few_count} classes:")
    print(f"     -id- ,", "-name-".center(longest_class_name_len), ", -count-")
    for class_id in few_classes_list:
        print(
            "    ",
            str(class_id).rjust(4),
            ",",
            str(classes.loc[classes.class_id == class_id].class_name.values[0]).rjust(
                longest_class_name_len
            ),
            ",",
            str(len(images.loc[images.class_id == class_id])).rjust(7),
        )

    # check alphabetical order
    class_names_in_alphabetical_order = is_list_in_alphabetical_order(
        class_names_in_class_id_order
    )

    # check that the difference between each class_id is 1
    class_ids_are_contiguous = all(
        [
            ((unique_class_ids[x] - unique_class_ids[x - 1]) == 1)
            for x in range(1, len(unique_class_ids))
        ]
    )

    print("    class names in alphabetical order:", class_names_in_alphabetical_order)
    print("    class ids are contiguous:", class_ids_are_contiguous)
    print("    shortest class name:", shortest_class_name)
    print("    longest class name:", longest_class_name)

    class_counts = list(images[images.class_id >= 0].class_id.value_counts().values)
    unique_class_id_counts = sorted(list(set(class_counts)))
    if len(unique_class_id_counts) == 1:
        print("    all classes are the same size:", unique_class_id_counts[0])
    else:
        print("    smallest class has size:", unique_class_id_counts[0])
        print("    largest class has size:", unique_class_id_counts[-1])
        print(
            "    average class size:",
            int(round(sum(class_counts) / len(class_counts), 0)),
        )

    print("#" * 70)
    # for now this is just informative
    return True


def find_folders_full_of_duplicates(images: DataFrame):
    """find all of the folders full of dupes"""
    # it temporarily creates a new column "folder", that is removed before returning

    # create a new column of just the folders with images in them
    images["folder"] = images.relative_path.apply(
        lambda x: "/".join(x.split("/")[:-1]) + "/"
    )

    # get a list of unique folders
    folders = sorted(list(images.folder.unique()), key=str.casefold)

    print("image count:", len(images), " folder count:", len(folders))
    if len(folders) == len(images):
        print("there is a folder for each image, skipping folder check")
        images = images.drop(["folder"], axis=1)
        return

    folders_that_are_full_of_dupes = []
    for folder in folders:
        hashes_inside = list(images[images.folder == folder].file_hash.unique())
        hashes_outside = list(images[images.folder != folder].file_hash.unique())
        if len(list(set(hashes_inside) - set(hashes_outside))) == 0:
            folders_that_are_full_of_dupes += [folder]

    if len(folders_that_are_full_of_dupes) == 0:
        print("there are no folders full of images that are duplicated elsewhere")
        images = images.drop(["folder"], axis=1)
        return

    # find which of those folders are also full of images marked problem=3
    folders_that_are_full_and_marked = []
    for folder in folders:
        count = len(images[(images.folder == folder) & (images.problem != 3)])
        if count == 0:
            folders_that_are_full_and_marked += [folder]

    if len(folders_that_are_full_and_marked) > 0:
        print(
            "\nthe following folders are full of images duplicated elsewhere AND marked problem=3"
        )
        for folder in folders_that_are_full_and_marked:
            print(f"  {folder}")

    folders_remaining = list(
        set(folders_that_are_full_of_dupes) - set(folders_that_are_full_and_marked)
    )
    if len(folders_remaining) > 0:
        print(
            "\nthe following folders are full of images duplicated elsewhere and NOT marked problem=3"
        )
        for folder in folders_remaining:
            print(f"  {folder}")

    print("")

    # remove folder column
    images = images.drop(["folder"], axis=1)


def check_if_all_archive_files_are_used(dataset_root_path: str):
    """check if all of the archive files are used,
    if an individual file doesn't show up at all, then it is assumed to not have been used"""

    chdir_with_create(dataset_root_path)

    if not os.path.isdir(dataset_root_path + "archives/"):
        print("no archives folder")
        return

    archive_files = os.listdir(dataset_root_path + "archives/")

    print("verify all archive files are used (this will be a manual check):")

    if len(archive_files) == 0:
        print("no archive files found")
        return

    archive_files = sorted(archive_files, key=str.casefold)

    if not os.path.isfile("prepare_dataset.py"):
        print("no prepare_dataset.py")
        return

    prepare_lines = None
    with open("prepare_dataset.py") as file_h:
        prepare_lines = file_h.readlines()

    assert not prepare_lines is None
    there_are_missing_files = False
    for file in archive_files:
        print("#" * 10 + "  " + file + "  " + "#" * 10)
        print("#" * 60)

        actual_line_count = 0
        for line in prepare_lines:
            if file in line:
                line = line.strip().replace("\n", "")
                print("  " + line)

                # if the line isn't a comment, then count it
                if line[0] != "#":
                    actual_line_count += 1

        print("#" * 60)
        if actual_line_count == 0:
            print("no (uncommented) lines found")
            there_are_missing_files = True

    if there_are_missing_files:
        print("there is at least one archive file never mentioned")
        return

    return


def validate_classes_dict(classes_dict: dict[str, int]) -> tuple[bool, list[int]]:
    """validate the classes_dict, and return a list of unique class_ids"""

    # since classes was stored as a dict, we already know that class_name will be unique
    print("validate classes dict")

    # at least 1 item
    if len(classes_dict) == 0:
        print("there are no classes provided")
        return (False, [])

    # check if all class_ids are integers
    non_int = [
        class_id
        for class_id in list(classes_dict.values())
        if not isinstance(class_id, int)
    ]
    if len(non_int) > 0:
        print("these class_ids are not integers:", non_int)
        return (False, [])

    # check if all class_ids >= 0
    too_low = [class_id for class_id in list(classes_dict.values()) if class_id < 0]
    if len(too_low) > 0:
        print("these class_ids are < 0:", too_low)
        return (False, [])

    # check if all of the class_ids are unique
    sorted_unique_id_list = sorted(list(set(list(classes_dict.values()))))
    if len(sorted_unique_id_list) != len(classes_dict):
        print("there are duplicate class_ids")
        for id in sorted_unique_id_list:
            count = list(classes_dict.values()).count(id)
            if count > 1:
                print(f"id {id} shows up {count} times")
        return (False, [])

    # check if all class_names are strings
    non_str = [
        class_name
        for class_name in list(classes_dict.keys())
        if not isinstance(class_name, str)
    ]
    if len(non_str) > 0:
        print("these class_names are not strings:", non_str)
        print([str(type(class_name)) for class_name in non_str])
        return (False, [])

    # check if all class_names are at least 1 character
    too_short = [
        class_name for class_name in list(classes_dict.keys()) if (len(class_name) == 0)
    ]
    if len(too_short) > 0:
        print("there are", len(too_short), "class_names that are empty")
        return (False, [])

    has_whitespace_on_ends = [
        class_name
        for class_name in list(classes_dict.keys())
        if len(class_name) != len(class_name.strip())
    ]
    if len(has_whitespace_on_ends) > 0:
        print(
            "these class_names have white space on either end:", has_whitespace_on_ends
        )
        return (False, [])

    encodes_differently = [
        class_name
        for class_name in list(classes_dict.keys())
        if len(class_name) != len(class_name.encode())
    ]
    if len(encodes_differently) > 0:
        print("these class_names have strange characters:", encodes_differently)
        print("this is just a warning")

    return (True, sorted_unique_id_list)


def validate_images_list(
    dataset_root_path: str, images_list: list[DatasetImage]
) -> tuple[bool, list[int]]:
    """validate the images list, and also return a list of class_ids seen"""

    print("validate images list")

    # at least 1 item
    if len(images_list) == 0:
        print("no images provided")
        return (False, [])

    list_of_relative_paths = []
    class_id_list = []
    for dataset_image in images_list:
        # this will validate everything it can
        if not dataset_image.validate(dataset_root_path):
            return (False, [])

        # collect the class_id (if >= 0)
        if dataset_image.class_id >= 0:
            if not dataset_image.class_id in class_id_list:
                class_id_list += [dataset_image.class_id]

        # collect the relative paths, so we can check if the files are unique
        # this is different from checking if the files contain the same data
        list_of_relative_paths += [dataset_image.relative_path]

    unique_path_count = len(list(set(list_of_relative_paths)))
    if unique_path_count != len(images_list):
        print("some of the files are duplicated")
        print("expected", len(images_list), "unique files, but got", unique_path_count)
        return (False, [])

    sorted_unique_id_list = sorted(list(set(class_id_list)))

    return (True, sorted_unique_id_list)


def validate_list_and_dict(
    dataset_root_path: str,
    images_list: list[DatasetImage],
    classes_dict: dict[str, int],
) -> bool:
    """this will do a first-pass validation on the results"""

    print("validate parsed dataset")

    ## check classes ##
    (result, class_id_list) = validate_classes_dict(classes_dict)
    if not result:
        print("failed during classes_dict checks")
        return False

    ## check images ##
    (result, class_ids_found_in_images_list) = validate_images_list(
        dataset_root_path, images_list
    )
    if not result:
        print("failed during images_list checks")
        return False

    # verify that the class_id sets are equal
    check_fwd = list(set(class_ids_found_in_images_list) - set(class_id_list))
    check_bkw = list(set(class_id_list) - set(class_ids_found_in_images_list))
    if (len(check_fwd) != 0) or (len(check_bkw) != 0):
        print("there are missing class_ids:")
        print("  missing id in classes_dict: ", check_fwd)
        print("  missing id in images_list: ", check_bkw)
        return False

    # loop through every image provided to look for a file in converted_files
    found_a_file_in_converted_files = False
    for dataset_image in images_list:
        # check if any of the images given in the list are inside converted_files
        if "converted_files/" in dataset_image.relative_path:
            found_a_file_in_converted_files = True
            break  # just exit out of loop once one is found

    # additional checks if converted_files folder exists
    if os.path.isdir(dataset_root_path + "converted_files/"):
        # check if at least one converted file was found in the list provided
        if not found_a_file_in_converted_files:
            print(
                "converted_files folder exists, but no images were provided from that folder"
            )
            return False

        # verify that all converted files are only of type png (independently)
        converted_files = get_all_files_in_directory(
            dataset_root_path + "converted_files/"
        )

        for converted_file in converted_files:
            extension = get_file_extension(converted_file)
            if extension != "png":
                print("only .png files are allowed in converted_files", converted_file)
                return False
    else:
        if found_a_file_in_converted_files:
            print(
                "no converted_files folder exists, but at least one image says it exists there"
            )
            return False

    # check for any additional folders inside root
    if not check_root_contents(dataset_root_path):
        return False

    return True


def check_root_contents(dataset_root_path: str) -> bool:
    """check root for any un-allowed contents"""

    # check for any additional files or folders inside root
    root_contents = os.listdir(dataset_root_path)

    # this should just be what is necessary to run prepare_dataset.py
    allowed_root_contents = [
        "archives",
        "original_files",
        "converted_files",
        "prepare_dataset.py",
        "dataset_utils.py",
        "__pycache__",  # cache is fine
        "examples",
    ]

    for item in root_contents:
        if not item in allowed_root_contents:
            print(item, "is not allowed in the dataset root path")
            return False

    return True


def get_all_files_in_directory(
    directory_path: str, remove_directory_path: str = ""
) -> list[str]:
    """this will return a list of all files in a directory
    Optionally, it can remove the path from the string"""

    # make sure there is a trailing slash
    assert directory_path[-1] == "/"
    assert os.path.isdir(directory_path)

    if len(remove_directory_path) > 0:
        # make sure that the provided remove_directory_path is the beginning of the directory_path
        assert directory_path[: len(remove_directory_path)] == remove_directory_path

        # modify the paths so that glob does the clean up automatically
        additional_directory = directory_path[len(remove_directory_path) :]
        directory_path = directory_path[: -len(additional_directory)]

        all_files_list = glob(
            additional_directory + "**/*", root_dir=directory_path, recursive=True
        )
    else:
        all_files_list = glob(directory_path + "**/*", recursive=True)

    # remove directories by testing all files if they are actually directories
    add_directory_path = directory_path if (len(remove_directory_path) > 0) else ""
    for i in reversed(range(len(all_files_list))):
        if os.path.isdir(add_directory_path + all_files_list[i]):
            del all_files_list[i]

    return all_files_list


def filter_file_list(list_of_files: list[str]) -> tuple[list[str], list[str]]:
    """this will return a tuple of the images files and non-image files
    it does not require that the list of files have the entire path
    """

    assert len(list_of_files) > 0

    image_files: list[str] = []
    non_image_files: list[str] = []

    for file_path in list_of_files:

        extension = get_file_extension(file_path)

        # validate the extension
        if extension in VALID_IMAGE_EXTENSIONS:
            image_files += [file_path]
        else:
            non_image_files += [file_path]

    suspect_non_image_files = []
    for file_path in non_image_files:
        # check each possible extension
        for extension in VALID_IMAGE_EXTENSIONS:
            # see if the extension shows up at all inside the filename (must force it to lowercase)
            if extension in file_path.lower():
                suspect_non_image_files += [file_path]
                break  # only need one match per filename

    if len(suspect_non_image_files) > 0:
        print("there are some suspect files that were marked as non_image_files")
        for file_path in suspect_non_image_files:
            print(file_path)

    return (image_files, non_image_files)


def get_file_extension(file_path: str) -> str:
    """returns the file extension"""

    assert len(file_path) > 0
    filename = os.path.basename(file_path)

    assert len(filename) > 0

    if "." in filename:
        extension = filename.split(".")[-1].lower()
    else:
        # extension could be empty
        extension = ""

    return extension


def fix_directory_permissions(directory_path: str):
    """fix all of the permissions for ugo+rX"""

    assert directory_path[-1] == "/"
    assert os.path.isdir(directory_path)

    # ugo means apply to all user/group/other
    # rX sets read for everything and x bit for directories only
    # -R is recursive
    run_cmd_get_output(["chmod", "ugo+rX", "-R", directory_path])


def get_hash_of_file(file_path: str) -> str:
    """this will return the xxh64sum hash for the given file"""

    # assert that the file is even a file
    assert os.path.isfile(file_path)

    # just extract the hash from the output
    # this would possibly have fewer collisions as md5sum.
    # however, for historical reasons, xxh64sum is used
    hash = run_cmd_get_output(["xxh64sum", file_path]).strip().split()[0]

    return hash


def get_hash_of_image(image_h: Image.Image) -> str:
    """this will return the md5sum hash for the given image"""

    # get the image_hash
    hashfunc = hashlib.md5()
    hashfunc.update(np.array(image_h))  # type: ignore
    return hashfunc.hexdigest()


def run_cmd_get_output(
    *popenargs: list, check: bool = True, capture_output: bool = True
) -> str:
    """run a subprocess command and return the utf-8 output"""

    result = subprocess.run(*popenargs, check=check, capture_output=capture_output).stdout.decode("utf-8")  # type: ignore

    return result


def is_image_valid(file_path: str, deeper_check: bool = False) -> bool:
    """this checks if the image is valid, according to PIL"""

    # assert that the file is even a file
    # inability to open the file, is not a failure of the hypothetical image
    assert os.path.isfile(file_path)

    # wrap in a try block to prevent failures from killing program
    try:
        with Image.open(file_path) as image_h:
            # this is the default verify function for PIL
            # it is not perfect, but will catch most issues
            # it raises an error if it finds an issue
            image_h.verify()

        # if also doing a deeper check
        if deeper_check:
            # must reopen file after doing verify
            with Image.open(file_path) as image_h:

                # assume a zero-sized dimension is invalid
                if (image_h.width == 0) or (image_h.height == 0):
                    return False

                if not Image.MAX_IMAGE_PIXELS is None:
                    if (image_h.width * image_h.height) > Image.MAX_IMAGE_PIXELS:
                        print(
                            file_path,
                            "is very large:",
                            image_h.width,
                            "x",
                            image_h.height,
                        )

        return True

    except:
        # failed trying to open or verify the file
        return False


def chdir_with_create(directory_path: str):
    """this will create a directory if necessary
    and then change to it"""

    assert directory_path[-1] == "/"

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

        # verify that it was created
        assert os.path.isdir(directory_path)

    os.chdir(directory_path)


def get_unique_directories_from_files_list(
    files_list: list[str], subdirectory_level: int
) -> list[str]:
    """get a list of the unique directories from a files list based on the subdirectory"""

    directories = []
    for image_path in files_list:
        directory = image_path.split("/")[subdirectory_level]
        if not directory in directories:
            directories += [directory]

    return directories


def create_examples_directory(dataset_root_path: str, example_count: int = 5):
    """this will load the sqlite and create an examples directory for every class"""
    print("create examples")

    chdir_with_create(dataset_root_path)

    # remove the examples directory first
    if not remove_folder_or_file_and_check("examples/"):
        sys.exit()

    os.mkdir(dataset_root_path + "examples/")
    assert os.path.isdir(dataset_root_path + "examples/")

    (df_images, df_classes) = get_dataframes_from_sqlite(
        dataset_root_path + IMAGES_DATABASE_FILE
    )

    # for each class...
    for row in df_classes.iterrows():
        class_name = row[1]["class_name"]
        class_id = row[1]["class_id"]

        # clean up class_name
        class_name = class_name.replace(" ", "_").replace("/", "_")
        folder_name = str(class_id) + "_" + class_name + "/"
        class_path = dataset_root_path + "examples/" + folder_name

        # get some examples
        examples = df_images.loc[df_images.class_id == class_id].relative_path

        sample_size = min(example_count, len(examples))
        # draw a few examples
        examples = random.sample(list(examples), k=sample_size)

        # create the folder
        chdir_with_create(class_path)

        # copy the images in
        for example in examples:
            original_path = dataset_root_path + example
            run_cmd_get_output(["cp", original_path, class_path])

    # make sure permissions are correct
    fix_directory_permissions(dataset_root_path + "examples/")


def remove_folder_or_file_and_check(path: str) -> bool:
    """remove the folder and verify"""

    removed = True

    if os.path.isdir(path):
        try:
            run_cmd_get_output(["rm", "-rf", path])
            removed = not os.path.isdir(path)
        except:
            removed = False
    elif os.path.isfile(path):
        try:
            run_cmd_get_output(["rm", path])
            removed = not os.path.isfile(path)
        except:
            removed = False

    if not removed:
        print(f"error removing {path}")

    # return True since the path is neither a file nor a folder
    return removed


def create_sqlite_only(dataset_root_path: str):
    """create the sqlite file using prepare_dataset.py"""

    print(f"###  create the {IMAGES_DATABASE_FILE} file only  ###")

    print("dir", dataset_root_path)

    if not os.path.isfile("prepare_dataset.py"):
        print("there is not prepare_dataset.py in this folder")
        return

    if not remove_folder_or_file_and_check("__pycache__"):
        return

    if not remove_folder_or_file_and_check("examples"):
        return

    if not remove_folder_or_file_and_check(IMAGES_DATABASE_FILE):
        return

    if not remove_folder_or_file_and_check("ilab.sqlite"):
        return

    # need to insert the path at the top to try to import the
    # correct prepare_dataset.py file
    sys.path.insert(0, dataset_root_path)

    try:
        # for the super datasets
        from prepare_dataset import create_subset_databases  # type: ignore

        create_subset_databases(dataset_root_path)

    except:
        # for most datasets
        from prepare_dataset import parse_dataset

        create_images_database(
            dataset_root_path,
            parse_dataset,
        )


def full_validation(dataset_root_path: str):
    """perform a full validation of the prepare_dataset.py pipeline"""

    print("###  full validation  ###")

    if not os.path.isfile("prepare_dataset.py"):
        print("there is not prepare_dataset.py in this folder")
        return

    print("-" * 50)
    print("prepare folder for validation")

    if not remove_folder_or_file_and_check("original_files"):
        print("(this is acceptable for the ilab datasets)")

    if not remove_folder_or_file_and_check("converted_files"):
        return

    if not remove_folder_or_file_and_check("__pycache__"):
        return

    if not remove_folder_or_file_and_check("examples"):
        return

    if not remove_folder_or_file_and_check(IMAGES_DATABASE_FILE):
        return

    if not remove_folder_or_file_and_check("ilab.sqlite"):
        return

    # check for any other disallowed contents
    if not check_root_contents(dataset_root_path):
        return

    print("-" * 50)
    print("now run prepare_dataset.py...")

    # execute without capturing the output
    try:
        subprocess.run(["python", "prepare_dataset.py"])
    except:
        pass

    if os.path.isfile(IMAGES_DATABASE_FILE):
        print(f"{IMAGES_DATABASE_FILE} successfully created")
        print("THIS DOES NOT MEAN THERE ARE NO ISSUES")
    else:
        print(f"{IMAGES_DATABASE_FILE} does not exist (this is okay for the supersets)")

    # remove the cache again (to make directory cleaner when done)
    remove_folder_or_file_and_check("__pycache__")


def get_dataframes_from_sqlite(sqlite_path: str) -> tuple[DataFrame, DataFrame]:
    """get the dataframes from the given sqlite file"""

    if not os.path.isfile(sqlite_path):
        print(sqlite_path, "does not exist")
        sys.exit(1)

    with sqlite3.connect(sqlite_path) as conn:
        df_images = pd.read_sql("select * from images", conn)
        df_classes = pd.read_sql("select * from classes", conn)

    return (df_images, df_classes)


if __name__ == "__main__":

    # there should be only one argument
    if len(sys.argv) < 2:
        print("an argument must be provided 'validate' or 'stats'")
        sys.exit()

    command = str(sys.argv[1])

    dataset_root_path = os.getcwd()
    # make sure path ends in a single trailing slash
    dataset_root_path = (dataset_root_path + "/").replace("//", "/")
    print("using path:", dataset_root_path)

    # commands that don't require sqlite file
    if command == "validate":
        full_validation(dataset_root_path)
        sys.exit()

    elif command == "sqlite":
        create_sqlite_only(dataset_root_path)
        sys.exit()

    elif command == "archive":
        check_if_all_archive_files_are_used(dataset_root_path)
        sys.exit()

    if not os.path.isfile(dataset_root_path + IMAGES_DATABASE_FILE):
        print(f"no {IMAGES_DATABASE_FILE}")
        sys.exit()
    (df_images, df_classes) = get_dataframes_from_sqlite(
        dataset_root_path + IMAGES_DATABASE_FILE
    )

    if command == "stats":
        calculate_statistics(dataset_root_path, df_images, df_classes)
        find_folders_full_of_duplicates(df_images)
        are_duplicates_real(df_images)
        problem_zero_dupes(df_images)

    elif command == "alpha":
        class_list = list(df_classes.class_name)  # type: ignore
        if is_list_in_alphabetical_order(class_list):
            print("classes are in alphabetical order")
        else:
            print("classes are NOT alphabetically sorted")

    elif command == "dupes":
        find_folders_full_of_duplicates(df_images)
        are_duplicates_real(df_images)
        problem_zero_dupes(df_images)
