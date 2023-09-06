"""this file will create the output sqlite files for each dataset

If will do the following:

A. An intra-dataset duplicate cleanup:
    (using the file hashes)
    1. Find all (problem=0) file hashes that show up in more than one class, 
       and for (problem=0) images with that file hash, set the problem to 100.
    2. Find all (problem=0) file hashes that still show up more than once,
       and for all but one of the (problem=0) images with that file hash, set the problem to 101.
    
    (using the image hashes)
    first, calculate images hashes for all remaining (problem=0) images
    3. Find all (problem=0) image hashes that show up in more than one class, 
       and for (problem=0) images with that image hash, set the problem to 102.
    4. Find all (problem=0) image hashes that still show up more than once,
       and for all but one of the (problem=0) images with that image hash, set the problem to 103.

B. An inter-dataset duplicate cleanup and removal of images that are too small:
    1. Find all (problem=0) file hashes that show up in more than one dataset,
       and for all (problem=0) images with that file hash, set the problem to 200.
    2. Find all (problem=0) image hashes that show up in more than one dataset,
       and for all (problem=0) images with that image hash, set the problem to 201.

C. Create train/val/test splits for each
    1. If any class_id contains fewer than 3 images, then set the problem to 300.
       There is a separate function for if the dataset contains subject_ids

"""

import math
import multiprocessing
import os
import sqlite3
import sys
import time

import dataset_utils
import pandas as pd
from pandas import DataFrame

root_directory = "/home2/tmp/u/DCT_files/"
main_directory = "0_collection"

MIN_CLASS_SIZE = 3  # for problem 300
SET_SPLITS = [80, 10, 10]  # will be converted into percents later

INPUT_DATABASE = "database.sqlite"
OUTPUT_DATABASE = "database_cleaned.sqlite"
COMBINED_DATABASE = "all_images.sqlite"

DATABASE_COLUMNS = [
    "relative_path",
    "class_id",
    "set_id",
    "problem",
    "subject_id",
    "file_size",
    "file_hash",
    "image_hash",
    "image_width",
    "image_height",
    "image_mode",
    "image_format",
]


def main():

    # first, get a list of the datasets
    datasets = os.listdir(root_directory)
    datasets = sorted(datasets, key=str.casefold)
    for i in reversed(range(len(datasets))):

        # remove misc files and directories
        if ".git" in datasets[i]:
            del datasets[i]
        elif main_directory == datasets[i]:
            del datasets[i]

        # remove the two superset directories
        elif "iNaturalist" == datasets[i]:
            del datasets[i]
        elif "Office-Home" == datasets[i]:
            del datasets[i]

    # initial checks
    print("first, check databases")
    verify_databases_for_datasets(datasets)

    #
    ####  A. intra-dataset pass  ####
    #

    print("A. intra-dataset pass:")
    print(f"create each dataset's {OUTPUT_DATABASE}:")
    t_start = time.perf_counter()
    print(f"begin using {dataset_utils.NUM_WORKERS} workers...  (will take time)")
    with multiprocessing.Pool(dataset_utils.NUM_WORKERS) as pool:
        # decrease chunksize because some datasets are huge
        pool.map(step_a, datasets, chunksize=1)

    print("done", round(time.perf_counter() - t_start), "sec")

    #
    ####  B. inter-dataset pass  ####
    #

    print("B. inter-dataset pass:")
    all_images = combine_all_sqlite_files(datasets)
    (dupe_file_hashes, dupe_image_hashes) = find_inter_dataset_dupes(all_images)

    t_start = time.perf_counter()

    print(f"update each dataset's {OUTPUT_DATABASE}:")
    print(f"begin using {dataset_utils.NUM_WORKERS} workers...  (will be fast)")
    with multiprocessing.Pool(dataset_utils.NUM_WORKERS) as pool:
        # multiple args, so use apply instead of map
        for directory in datasets:
            pool.apply_async(
                step_b,
                (
                    directory,
                    dupe_file_hashes,
                    dupe_image_hashes,
                ),
            )

        # begin the work (outside loop)
        pool.close()
        pool.join()

    print("done", round(time.perf_counter() - t_start), "sec")

    #
    ####  C. create train/val/test splits  ####
    #

    print("C. create train/val/test splits:")
    create_set_splits(datasets)

    # final save
    print("finally, save a combined file of all images")
    combine_all_sqlite_files(datasets, COMBINED_DATABASE)


#
## initial checks
#


def verify_databases_for_datasets(datasets: list[str]):
    for directory in datasets:
        if not os.path.isfile(root_directory + directory + "/" + INPUT_DATABASE):
            print(f"no {INPUT_DATABASE} file for: {directory}")
            sys.exit()

    with multiprocessing.Pool(dataset_utils.NUM_WORKERS) as pool:
        # decrease chunksize because some datasets are huge
        results = pool.map(verify_columns_for_database, datasets, chunksize=1)

    if not all(results):
        print(f"some {INPUT_DATABASE} files have issues:")
        for i, v in enumerate(results):
            if not v:
                print(datasets[i])
        sys.exit()


def verify_columns_for_database(directory: str) -> bool:
    """verify every sqlite file contains the correct columns"""

    # move to the directory
    dataset_utils.chdir_with_create(root_directory + directory + "/")

    (images, _) = dataset_utils.get_dataframes_from_sqlite(INPUT_DATABASE)

    # check order of keys and correct version of sqlite
    return list(images.keys()) == DATABASE_COLUMNS


#
# A. Do the intra-dataset duplicate clean up, and create the output sqlite file
#


# helper function that is used elsewhere
def get_index(f):
    return f.index[0]


def step_a(directory: str) -> None:
    """this performs all of step a, and saving the sqlite file at the end"""

    # move to the directory
    dataset_utils.chdir_with_create(root_directory + directory + "/")

    (images, classes) = dataset_utils.get_dataframes_from_sqlite(INPUT_DATABASE)

    # check order of keys and correct version of sqlite
    assert list(images.keys()) == DATABASE_COLUMNS

    #
    # A.1 clean up the hashes with more than one class
    #

    # find all hashes with more than one class id
    (dupe_hashes_with_more_than_one_class_id, _) = dataset_utils.problem_zero_dupes(
        images, silent=True
    )
    # for all images with this issue, that don't have a problem, set to problem=100
    images.loc[
        (images.file_hash.isin(dupe_hashes_with_more_than_one_class_id))
        & (images.problem == 0),
        "problem",
    ] = 100

    del dupe_hashes_with_more_than_one_class_id

    #
    # A.2 clean up the duplicate hashes that remain (keep only one image of each hash)
    #

    # find the remaining problem=0 duplicate hashes
    dupe_hashes = images[images.problem == 0].file_hash.value_counts()
    # keep only those that are actual duplicates (2+ images with same hash)
    dupe_hashes = dupe_hashes[dupe_hashes > 1]
    # just keep the list
    dupe_hashes = list(dupe_hashes.index)

    # only check if there are still dupe hashes remaining
    if len(dupe_hashes) > 0:
        # for each hash in that list

        # group by the hash and relative_path
        # the apply gets the index of the pair (always unique)
        # set the fill_value to -1 to make it easy to filter
        # and return a numpy matrix
        results = (
            images.loc[(images.file_hash.isin(dupe_hashes)) & (images.problem == 0)]
            .groupby(["file_hash", "relative_path"])
            .apply(get_index)
            .unstack(fill_value=-1)
            .to_numpy()
        )

        for i in range(results.shape[0]):
            # get only the real indexes for the row
            indexes = [x for x in list(results[i][:]) if x >= 0]

            # now, sort those by the relative path and return the new list
            sorted_indexes = list(
                images.loc[indexes]
                .sort_values(by="relative_path", key=lambda col: col.str.casefold())
                .index
            )

            assert len(sorted_indexes) > 1

            # the first index will be left alone, the others will get problem=101
            images.loc[sorted_indexes[1:], "problem"] = 101

            del indexes, sorted_indexes

        del results
    del dupe_hashes

    #
    # A.3 clean up the image hashes in more than one class
    #

    dupe_image_hashes = images[images.problem == 0].image_hash.value_counts()
    # keep only those that are actual duplicates
    dupe_image_hashes = dupe_image_hashes[dupe_image_hashes > 1]
    # just keep the list
    dupe_image_hashes = list(dupe_image_hashes.index)

    if len(dupe_image_hashes) > 0:
        # group by hash and class_id, and do the following:
        # 1) count the number of images with each combination: .size()
        # 2) if there is more than 0 images for a combination, set the value as a 1: .gt(0).mul(1)
        # 3) unstack the combos, and wherever there wasn't a pair, set the value as 0: .unstack(fill_value=0)
        results = (
            images[(images.image_hash.isin(dupe_image_hashes)) & (images.problem == 0)]
            .groupby(["image_hash", "class_id"])
            .size()
            .gt(0)
            .mul(1)
            .unstack(fill_value=0)
        )

        # 4) sum the results along the rows (this will count the number of hash/class_id pairs for each hash)
        results = results.sum(axis=1)
        # 5) keep only the hashes that have at least 2 class_id pairs
        results = results[results > 1]

        dupe_image_hashes_with_more_than_one_class_id = list(results.index)

        # for all images with this issue, that don't have a problem, set to problem=102
        images.loc[
            (images.image_hash.isin(dupe_image_hashes_with_more_than_one_class_id))
            & (images.problem == 0),
            "problem",
        ] = 102

        del results
    del dupe_image_hashes

    #
    # A.4 clean up the remaining duplicate image hashes
    #

    # find the remaining problem=0 duplicate hashes
    dupe_image_hashes = images[images.problem == 0].image_hash.value_counts()
    # keep only those that are actual duplicates (2+ images with same hash)
    dupe_image_hashes = dupe_image_hashes[dupe_image_hashes > 1]
    # just keep the list
    dupe_image_hashes = list(dupe_image_hashes.index)

    # only check if there are still dupe hashes remaining
    if len(dupe_image_hashes) > 0:
        # for each hash in that list

        # group by the hash and relative_path
        # the apply gets the index of the pair (always unique)
        # set the fill_value to -1 to make it easy to filter
        # and return a numpy matrix
        results = (
            images.loc[
                (images.image_hash.isin(dupe_image_hashes)) & (images.problem == 0)
            ]
            .groupby(["image_hash", "relative_path"])
            .apply(get_index)
            .unstack(fill_value=-1)
            .to_numpy()
        )

        for i in range(results.shape[0]):
            # get only the real indexes for the row
            indexes = [x for x in list(results[i][:]) if x >= 0]

            # now, sort those by the relative path and return the new list
            sorted_indexes = list(
                images.loc[indexes]
                .sort_values(by="relative_path", key=lambda col: col.str.casefold())
                .index
            )

            assert len(sorted_indexes) > 1

            # the first index will be left alone, the others will get problem=103
            images.loc[sorted_indexes[1:], "problem"] = 103

            del indexes, sorted_indexes

        del results
    del dupe_image_hashes

    # make sure to update set_id = -1 and class_id = -1
    images.loc[images.problem > 0, "class_id"] = -1
    images.loc[images.problem > 0, "set_id"] = -1

    # remove the file, since it is being replaced
    if os.path.isfile(OUTPUT_DATABASE):
        os.remove(OUTPUT_DATABASE)

    # save to disk
    with sqlite3.connect(OUTPUT_DATABASE) as conn:
        images.to_sql("images", conn, if_exists="replace", index=False)
        classes.to_sql("classes", conn, if_exists="replace", index=False)
        conn.commit()

    # erase these to catch any errors
    del images, classes, conn


#
# B. Do the inter-dataset duplicate clean up
#


def get_images_from_sqlite(directory: str) -> DataFrame:
    # move to the directory
    dataset_utils.chdir_with_create(root_directory + directory + "/")

    # load just the images
    (images, _) = dataset_utils.get_dataframes_from_sqlite(OUTPUT_DATABASE)

    assert list(images.keys()) == DATABASE_COLUMNS

    # store the name of the dataset (could be useful)
    images["dataset"] = directory

    return images


def combine_all_sqlite_files(
    datasets: list[str], save_file_name: str = ""
) -> DataFrame:

    new_columns = DATABASE_COLUMNS.copy()
    new_columns.append("dataset")
    all_images = pd.DataFrame(columns=new_columns)

    t_start = time.perf_counter()
    print("build a master dataframe of all images")

    print(f"begin using {dataset_utils.NUM_WORKERS} workers...")
    with multiprocessing.Pool(dataset_utils.NUM_WORKERS) as pool:
        # decrease chunksize because some datasets are huge
        list_of_dataframes = pool.map(get_images_from_sqlite, datasets, chunksize=1)

    # concat, ignoring the index
    all_images: DataFrame = pd.concat(list_of_dataframes, axis=0, ignore_index=True)

    print("done", round(time.perf_counter() - t_start), "sec")

    if len(save_file_name) > 0:
        print(f"save to file {save_file_name}")
        with sqlite3.connect(
            root_directory + main_directory + "/" + save_file_name
        ) as conn:
            all_images.to_sql("images", conn, if_exists="replace", index=False)
            conn.commit()

    return all_images


def find_inter_dataset_dupes(all_images: DataFrame) -> tuple[list[str], list[str]]:

    print("find inter-dataset dupes (from file and image hashes)")
    # find the inter-dataset duplicate hashes
    dupe_hashes = all_images[all_images.problem == 0].file_hash.value_counts()
    # keep only those that are actual duplicates (2+ images with same hash)
    dupe_hashes = dupe_hashes[dupe_hashes > 1]
    # just keep the list
    dupe_hashes = list(dupe_hashes.index)

    print("  inter-dataset dupe stats (file hash):")
    print("    hash count:", len(dupe_hashes))
    print(
        "    image count:",
        len(
            all_images[
                (all_images.problem == 0) & all_images.file_hash.isin(dupe_hashes)
            ]
        ),
    )

    # find the inter-dataset duplicate hashes
    dupe_image_hashes = all_images[
        (all_images.problem == 0) & (all_images.image_hash != "")
    ].image_hash.value_counts()
    # keep only those that are actual duplicates (2+ images with same hash)
    dupe_image_hashes = dupe_image_hashes[dupe_image_hashes > 1]
    # just keep the list
    dupe_image_hashes = list(dupe_image_hashes.index)

    print("  inter-dataset dupe stats (image hash):")
    print("    hash count:", len(dupe_image_hashes))
    print(
        "    image count:",
        len(
            all_images[
                (all_images.problem == 0)
                & all_images.image_hash.isin(dupe_image_hashes)
            ]
        ),
    )

    return (dupe_hashes, dupe_image_hashes)


def step_b(
    directory: str, dupe_file_hashes: list[str], dupe_image_hashes: list[str]
) -> None:
    """this performs all of step b, and saving the sqlite file at the end"""

    # move to the directory
    dataset_utils.chdir_with_create(root_directory + directory + "/")

    # load the output database
    (images, classes) = dataset_utils.get_dataframes_from_sqlite(OUTPUT_DATABASE)

    #
    # B.1 cross-dataset duplicate (based on file hash)
    #

    # if cross-dataset dupe (and problem=0), set problem 200
    images.loc[
        (images.file_hash.isin(dupe_file_hashes)) & (images.problem == 0),
        "problem",
    ] = 200

    #
    # B.2 cross-dataset duplicate (based on image hash)
    #

    # if cross-dataset dupe (and problem=0), set problem 201
    images.loc[
        (images.image_hash.isin(dupe_image_hashes)) & (images.problem == 0),
        "problem",
    ] = 201

    # make sure to update set_id = -1 and class_id = -1
    images.loc[images.problem > 0, "class_id"] = -1
    images.loc[images.problem > 0, "set_id"] = -1

    # remove the file, since it is being replaced
    if os.path.isfile(OUTPUT_DATABASE):
        os.remove(OUTPUT_DATABASE)

    # save to disk
    with sqlite3.connect(OUTPUT_DATABASE) as conn:
        images.to_sql("images", conn, if_exists="replace", index=False)
        classes.to_sql("classes", conn, if_exists="replace", index=False)
        conn.commit()


#
# C. Create the train/val/test splits
#


def create_set_splits(datasets: list[str]):

    # calculate the percents relative to total
    total_value = sum(SET_SPLITS)
    for set_id in range(3):
        SET_SPLITS[set_id] = SET_SPLITS[set_id] / total_value  # type: ignore

    print(f"using set percents: {SET_SPLITS}")

    t_start = time.perf_counter()
    print(f"begin using {dataset_utils.NUM_WORKERS} workers...")
    with multiprocessing.Pool(dataset_utils.NUM_WORKERS) as pool:
        # decrease chunksize because some datasets are huge
        pool.map(step_c, datasets, chunksize=1)

    print("done", round(time.perf_counter() - t_start), "sec")


def val_or_test_size(percent: float, total: int) -> int:
    """calculate the validation or test set size"""

    # round up, and minimum size is 1
    return max(1, math.ceil(percent * total))


def get_set_sizes(total: int) -> tuple[int, int, int]:
    """calculate train/val/test size given a total"""

    val_size = val_or_test_size(SET_SPLITS[1], total)
    test_size = val_or_test_size(SET_SPLITS[2], total)
    train_size = total - (val_size + test_size)

    return (train_size, val_size, test_size)


def step_c(directory: str):

    # move to the directory
    dataset_utils.chdir_with_create(root_directory + directory + "/")

    # load the output database
    (images, classes) = dataset_utils.get_dataframes_from_sqlite(OUTPUT_DATABASE)

    # sort by file_hash, which will give a repeatable random order
    # (does not depend on a seed or algo)
    # note: for images with the same hash, they have problems,
    # so there will be unique hashes for each image
    images = images.sort_values(["file_hash"]).reset_index(drop=True)

    # if there are subject_ids, then process differently
    if len(images[(images.problem == 0) & (images.subject_id != -1)]) > 0:

        # images is modified in-place
        step_c_with_subject_ids(images)

    else:

        # else, there are no subject ids, and the sets can be created normally

        # for all images without problems, set the set_id to 0 (training)
        images.loc[images.problem == 0, "set_id"] = 0

        # get all of the class_ids, sorted
        class_id_list = sorted(list(images.loc[images.problem == 0].class_id.unique()))

        for class_id in class_id_list:
            # get all of the indexes for a given class_id
            indexes_for_class_id = images.loc[
                (images.problem == 0) & (images.class_id == class_id)
            ].index

            # if there are fewer than MIN_CLASS_SIZE images,
            # then change problem to 300 and class_id/set_id to -1
            if len(indexes_for_class_id) < MIN_CLASS_SIZE:
                images.loc[indexes_for_class_id, "problem"] = 300
                images.loc[indexes_for_class_id, "class_id"] = -1
                images.loc[indexes_for_class_id, "set_id"] = -1
                continue  # and move to next class_id

            # get the val/test size
            (_, val_size, test_size) = get_set_sizes(len(indexes_for_class_id))

            # take the val indexes as the first set of those
            val_set_indexes = indexes_for_class_id[:val_size]
            images.loc[val_set_indexes, "set_id"] = 1

            # take the test indexes as the second set of those
            test_set_indexes = indexes_for_class_id[val_size : (val_size + test_size)]
            images.loc[test_set_indexes, "set_id"] = 2

            # note: the remaining images already have a set_id of 0 (training)

    # make sure everything got a set
    assert len(images[(images.problem == 0) & (~images.set_id.isin([0, 1, 2]))]) == 0

    # make sure there are images in each set
    for set_id in range(3):
        assert len(images[(images.problem == 0) & (images.set_id == set_id)]) > 0

    # show set/class results
    # print(
    #     images.loc[(images.problem == 0) & (images.set_id >= 0)]
    #     .groupby(["set_id", "class_id"])
    #     .size()
    #     .unstack(fill_value=0)
    # )

    # remove the file, since it is being replaced
    if os.path.isfile(OUTPUT_DATABASE):
        os.remove(OUTPUT_DATABASE)

    # save to disk
    with sqlite3.connect(OUTPUT_DATABASE) as conn:
        images.to_sql("images", conn, if_exists="replace", index=False)
        classes.to_sql("classes", conn, if_exists="replace", index=False)
        conn.commit()


## support functions for set creation with subject ids
def create_zeroed_dict(key_list: list[int]) -> dict[int, int]:
    """create a new dict with the same keys, but values=0"""
    output_dict = {}
    for class_id in key_list:
        output_dict.update({class_id: 0})
    return output_dict


def add_dict_values(
    main_dict: dict[int, int], add_dict: dict[int, int], multiply_value: int = 1
):
    """add the values from the add_dict to the main_dict,
    it also allows for subtracting values"""

    for class_id in list(main_dict.keys()):
        if class_id in add_dict:
            main_dict[class_id] = main_dict[class_id] + (
                multiply_value * add_dict[class_id]
            )


def all_dict_values_lte(
    check_dict: dict[int, int], desired_dict: dict[int, int]
) -> bool:
    """check that each key value is <= the desired value"""
    for class_id in list(desired_dict.keys()):
        if check_dict[class_id] > desired_dict[class_id]:
            return False
    return True


def which_set_is_most_behind(
    current_set_class_totals: list[dict[int, int]], final_set_totals: list[int]
) -> int:
    """return which set is most behind desired percent"""

    # get the % diffs
    sets_per_diff: dict[int, float] = {}
    for set_id in range(3):
        current_overall_total = sum(current_set_class_totals[set_id].values())
        per_diff = (
            current_overall_total - final_set_totals[set_id]
        ) / final_set_totals[set_id]
        sets_per_diff.update({set_id: per_diff})

    # sort the sets by increasing value
    sorted_sets = sorted(sets_per_diff.items(), key=lambda x: x[1])
    # take the first item
    most_behind = sorted_sets[0]
    # make sure the value is negative
    assert most_behind[1] < 0

    # return the set_id
    return most_behind[0]


# main function for set creation with subject ids
def step_c_with_subject_ids(images: DataFrame):
    """this is for creating the splits when there are subject ids"""

    assert len(images[(images.problem == 0) & (images.subject_id != -1)]) > 0

    # set all of the images to a fake set, so that assigned subjects can be tracked
    # -2 was not allowed by the validator, so no images will currently be assigned to it
    images.loc[images.problem == 0, "set_id"] = -2

    # get the entire list of subject_ids
    subject_id_list = sorted(list(images.loc[images.problem == 0].subject_id.unique()))
    class_id_list = sorted(list(images.loc[images.problem == 0].class_id.unique()))

    # get the class_id counts for every subject
    # instead of calculating it twice
    class_counts_for_subjects = (
        images.loc[images.problem == 0]
        .groupby(["class_id", "subject_id"])
        .size()
        .unstack(fill_value=0)
    )

    # get the totals for each class id
    class_id_totals: dict[int, int] = (
        images.loc[images.problem == 0]
        .class_id.value_counts()
        .sort_index()
        .to_dict()  # type: ignore
    )

    update_class_totals = False
    for class_id in class_id_list:
        # if any of the classes are too small, then eliminate them
        if class_id_totals[class_id] < MIN_CLASS_SIZE:
            update_class_totals = True
            indexes = images.loc[
                (images.problem == 0) & (images.class_id == class_id)
            ].index
            images.loc[indexes, "problem"] = 300
            images.loc[indexes, "class_id"] = -1
            images.loc[indexes, "set_id"] = -1

    if update_class_totals:
        # update the totals again
        class_id_totals: dict[int, int] = (
            images.loc[images.problem == 0]
            .class_id.value_counts()
            .sort_index()
            .to_dict()  # type: ignore
        )

    # make sure no -1 class_ids were captured
    assert not -1 in list(class_id_totals.keys())

    # get sizes for each set
    final_set_class_totals: list[dict[int, int]] = []
    for set_id in range(3):
        final_set_class_totals += [{}]

    for class_id in class_id_list:
        (train_size, val_size, test_size) = get_set_sizes(class_id_totals[class_id])

        final_set_class_totals[0].update({class_id: train_size})
        final_set_class_totals[1].update({class_id: val_size})
        final_set_class_totals[2].update({class_id: test_size})

    # get the overall totals
    final_set_totals: list[int] = []
    for set_id in range(3):
        final_set_totals += [sum(final_set_class_totals[set_id].values())]

    # create an empty dict (key is subject_id and value is set_id)
    subject_set_assignment_dict: dict[int, int] = {}

    # create current counts so that no lookups are necessary
    current_set_class_totals: list[dict[int, int]] = []
    for set_id in range(3):
        current_set_class_totals += [create_zeroed_dict(class_id_list)]

    # if there are no odd counts, then it is impossible to fill any set
    # that is an odd size, so check if there are odd counts or not
    unique_subject_image_counts = list(
        images.loc[images.problem == 0].subject_id.value_counts().unique()
    )
    has_odd_counts = any([(x % 2) != 0 for x in unique_subject_image_counts])

    if not has_odd_counts:
        # if all counts are even, then it is always possible to make all sets be even

        # for each class, check if the totals are odd, and adjust by
        # moving one image from the training set to the val/test set
        for class_id in class_id_list:
            # check in val set
            if (final_set_class_totals[1][class_id] % 2) == 1:
                final_set_class_totals[1][class_id] += 1
                final_set_class_totals[0][class_id] -= 1

            # check in test set
            if (final_set_class_totals[2][class_id] % 2) == 1:
                final_set_class_totals[2][class_id] += 1
                final_set_class_totals[0][class_id] -= 1

        # update the overall counts
        # and verify that the totals are correct
        for set_id in range(3):
            final_set_totals[set_id] = sum(final_set_class_totals[set_id].values())
            assert (final_set_totals[set_id] % 2) == 0

    # helper functions that rely rely on local variables
    def place_subject_into_set(
        subject_id: int, set_id: int, possible_addition: dict[int, int]
    ) -> bool:
        """try to place subject into set, return True if successful"""

        assert subject_id >= 0
        assert set_id in [0, 1, 2]

        # copy the current totals
        possible_class_totals = current_set_class_totals[set_id].copy()

        # add them together
        add_dict_values(possible_class_totals, possible_addition)

        # now, test if any of the class_id counts are too large
        if not all_dict_values_lte(
            possible_class_totals, final_set_class_totals[set_id]
        ):
            return False

        # otherwise, add it to the subject dict
        subject_set_assignment_dict.update({subject_id: set_id})

        # and set the set_id
        images.loc[
            (images.problem == 0) & (images.subject_id == subject_id), "set_id"
        ] = set_id

        # update the totals
        add_dict_values(current_set_class_totals[set_id], possible_addition)

        # success
        return True

    def find_a_subject_to_remove(problematic_subject_id: int) -> int:
        class_id_counts = class_counts_for_subjects[problematic_subject_id]

        # this code works best if only one class is at play, but not required
        # get the class that has the problem
        problematic_class_id: int = class_id_counts[class_id_counts > 0].index[0]  # type: ignore
        problematic_class_size: int = class_id_counts[class_id_counts > 0].values[0]  # type: ignore

        # placeholder values
        smallest_fix_count = 10000000
        smallest_fix_set = -1

        for set_id in range(3):
            current_class_size = current_set_class_totals[set_id][problematic_class_id]
            final_class_size = final_set_class_totals[set_id][problematic_class_id]

            fix_count = (current_class_size + problematic_class_size) - final_class_size

            if (fix_count < smallest_fix_count) and (fix_count > 0):
                smallest_fix_count = fix_count
                smallest_fix_set = set_id

        assert smallest_fix_set != -1

        # work backwards
        reverse_order_of_addition = list(
            reversed(list(subject_set_assignment_dict.keys()))
        )

        subject_id_to_remove = -1
        for subject_id in reverse_order_of_addition:
            results = class_counts_for_subjects[subject_id]

            # find a subject that has the smallest_fix_count in the
            # problematic class  and also is currently assigned to the
            # set where that could work
            if (results[problematic_class_id] == smallest_fix_count) and (  # type: ignore
                subject_set_assignment_dict[subject_id] == smallest_fix_set
            ):
                subject_id_to_remove = subject_id
                break

        # verify that a subject was found
        assert subject_id_to_remove > -1

        return subject_id_to_remove

    def distribute_all_subjects(
        subjects_to_distribute_list: list[int], class_id_counts: DataFrame
    ):
        """take a list of subjects and distributed them to the sets"""

        # flags used to indicate progress
        desired_set_id = -1
        failures_in_a_row = 0

        # loop while the subjects_to_distribute_list contains subjects
        while len(subjects_to_distribute_list) > 0:

            current_subject = subjects_to_distribute_list[0]

            # check if it desired to calculate the best
            if desired_set_id == -1:
                desired_set_id = which_set_is_most_behind(
                    current_set_class_totals, final_set_totals
                )

            # to start off, the subject that appears first will go into the training set
            counts_for_current_subject: dict[int, int] = class_id_counts[
                current_subject
            ].to_dict()  # type: ignore

            result = place_subject_into_set(
                current_subject, desired_set_id, counts_for_current_subject
            )

            if not result:
                # if it failed, just go to the next automatically
                desired_set_id = (desired_set_id + 1) % 3
                failures_in_a_row += 1

                # if it tried all 3 sets, then try to find a subject to remove from set
                if failures_in_a_row > 2:

                    subject_to_remove = find_a_subject_to_remove(current_subject)

                    # get the set
                    set_id_opening = subject_set_assignment_dict[subject_to_remove]

                    # remove the subject from the list of assignments
                    del subject_set_assignment_dict[subject_to_remove]

                    # add the subject back to the beginning of the list to distribute
                    subjects_to_distribute_list.insert(0, subject_to_remove)

                    # remove the class counts for the subject
                    class_counts_for_removed_subject: dict[int, int] = class_id_counts[
                        subject_to_remove
                    ].to_dict()  # type: ignore

                    add_dict_values(
                        current_set_class_totals[set_id_opening],
                        class_counts_for_removed_subject,
                        -1,
                    )

                    # fix the database too
                    images.loc[
                        (images.problem == 0)
                        & (images.subject_id == subject_to_remove),
                        "set_id",
                    ] = -2

                    # now, try to add the current subject to the same set
                    result = place_subject_into_set(
                        current_subject, set_id_opening, counts_for_current_subject
                    )

                    # if successful
                    if result:
                        subjects_to_distribute_list.remove(current_subject)
                        failures_in_a_row = 0
                        desired_set_id = -1  # set to find best
                    else:
                        print("failed again, exiting")
                        break

            else:
                subjects_to_distribute_list.remove(current_subject)
                failures_in_a_row = 0
                desired_set_id = -1  # set to find best

    #
    ### work on the subjects in more than one class
    #

    # get the number of classes that each subject shows up in, in descending order
    class_counts_for_each_subject_id = (
        images.loc[images.problem == 0]
        .groupby(["subject_id", "class_id"])
        .size()
        .gt(0)
        .mul(1)
        .unstack(fill_value=0)
        .apply(lambda x: x.sum(), axis=1)
        .sort_values(ascending=False)  # type: ignore
    )
    subjects_in_more_than_one_class = list(
        (class_counts_for_each_subject_id[class_counts_for_each_subject_id > 1]).index
    )

    if len(subjects_in_more_than_one_class) > 0:
        distribute_all_subjects(
            subjects_in_more_than_one_class, class_counts_for_subjects
        )

    #
    ### now, work on the remaining subjects (all have only one class)
    #

    # get the subjects that haven't been assigned to a set yet (set_id=-2)
    subjects_in_one_class = (
        images.loc[(images.problem == 0) & (images.set_id == -2)]
        .groupby(["subject_id", "class_id"])
        .size()
        .gt(0)
        .mul(1)
        .unstack(fill_value=0)
        .apply(lambda x: x.sum(), axis=1)
        .sort_values(ascending=False)  # type: ignore
    )

    # make sure that they only appear in one class
    assert max(subjects_in_one_class.values) == 1
    subjects_in_one_class = list(subjects_in_one_class.index)

    # now, get the sorted list by number of images (descending order of images)
    subjects_in_one_class = images.loc[
        (images.problem == 0) & (images.subject_id.isin(subjects_in_one_class))
    ].subject_id.value_counts()
    subjects_in_one_class = list(subjects_in_one_class.index)

    if len(subjects_in_one_class) > 0:
        distribute_all_subjects(subjects_in_one_class, class_counts_for_subjects)

    # check that all images were assigned to a set
    assert len(images.loc[images.set_id == -2]) == 0

    # check that every subject shows up in only one set
    set_counts_per_subject = (
        images.loc[images.problem == 0]
        .groupby(["subject_id", "set_id"])
        .size()
        .gt(0)
        .mul(1)
        .unstack(fill_value=0)
        .apply(lambda x: x.sum(), axis=1)
        .sort_values(ascending=False)  # type: ignore
    )
    assert set(set_counts_per_subject.values) == set([1])

    # make sure every subject is accounted for
    assert set(subject_id_list) == set(subject_set_assignment_dict.keys())

    # return to main step_c function


# called from terminal
if __name__ == "__main__":
    main()
