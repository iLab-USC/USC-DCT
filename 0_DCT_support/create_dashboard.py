import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass

import dataset_utils
import pandas as pd

root_directory = "/home2/tmp/u/DCT_files/"

which_database_to_use = "ilab.sqlite"
print("using: ", which_database_to_use)

directories = os.listdir(root_directory)
directories = sorted(directories, key=str.casefold)
for i in reversed(range(len(directories))):

    # remove misc files and directories
    if ".git" in directories[i]:
        del directories[i]
    if "0_collection" in directories[i]:
        del directories[i]

    # remove the two superset directories
    if "iNaturalist" == directories[i]:
        del directories[i]
    if "Office-Home" == directories[i]:
        del directories[i]


@dataclass()
class DatasetStats(object):
    name: str
    classes: int = -1
    classes_used: int = -1
    p_0: int = 0
    p_1: int = 0
    p_2: int = 0
    p_3: int = 0
    p_4: int = 0
    p_5: int = 0
    p_6: int = 0
    p_7: int = 0
    p_8: int = 0
    p_100: int = 0
    p_101: int = 0
    p_102: int = 0
    p_103: int = 0
    p_200: int = 0
    p_201: int = 0
    p_300: int = 0
    s_n1: int = -1
    s_0: int = -1
    s_1: int = -1
    s_2: int = -1
    set_prob_match: str = ""
    p_3_are_dupes: str = ""
    p_0_unique_hashes: int = 0
    p_0_dupes: int = 0
    class_alpha: str = ""
    class_gap: str = ""
    p_0_dupes_with_gt_one_class: int = 0
    p_0_dupes_with_gt_count: int = 0


datasets = []
for directory in directories:
    print(directory)
    dataset_stats = DatasetStats(directory)

    # make sure it is a directory with prepare_dataset.py
    if os.path.isfile(root_directory + directory + "/" + which_database_to_use):
        dataset_utils.chdir_with_create(root_directory + directory + "/")

        if os.path.isfile(which_database_to_use):

            (images, classes) = dataset_utils.get_dataframes_from_sqlite(
                which_database_to_use
            )

            assert "file_hash" in list(images.keys())

            dataset_stats.classes = len(classes)
            dataset_stats.classes_used = len(
                images[images.class_id >= 0].class_id.unique()
            )

            problem_sum = 0
            for problem_i in [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                100,
                101,
                102,
                103,
                200,
                201,
                300,
            ]:
                count = len(images.loc[images.problem == problem_i])
                if problem_i == 0:
                    dataset_stats.p_0 = count
                elif problem_i == 1:
                    dataset_stats.p_1 = count
                elif problem_i == 2:
                    dataset_stats.p_2 = count
                elif problem_i == 3:
                    dataset_stats.p_3 = count
                elif problem_i == 4:
                    dataset_stats.p_4 = count
                elif problem_i == 5:
                    dataset_stats.p_5 = count
                elif problem_i == 6:
                    dataset_stats.p_6 = count
                elif problem_i == 7:
                    dataset_stats.p_7 = count
                elif problem_i == 8:
                    dataset_stats.p_8 = count
                elif problem_i == 100:
                    dataset_stats.p_100 = count
                elif problem_i == 101:
                    dataset_stats.p_101 = count
                elif problem_i == 102:
                    dataset_stats.p_102 = count
                elif problem_i == 103:
                    dataset_stats.p_103 = count
                elif problem_i == 200:
                    dataset_stats.p_200 = count
                elif problem_i == 201:
                    dataset_stats.p_201 = count
                elif problem_i == 300:
                    dataset_stats.p_300 = count

                if problem_i > 0:
                    problem_sum += count

            set_problem_sum = 0
            for set_id in [-1, 0, 1, 2]:
                count = len(images.loc[images.set_id == set_id])

                if set_id == -1:
                    dataset_stats.s_n1 = count
                elif set_id == 0:
                    dataset_stats.s_0 = count
                elif set_id == 1:
                    dataset_stats.s_1 = count
                elif set_id == 2:
                    dataset_stats.s_2 = count

                if set_id == -1:
                    set_problem_sum = count

            set_problem_correct = "yes" if (problem_sum == set_problem_sum) else "no"
            dataset_stats.set_prob_match = set_problem_correct

            duplicate_hashes = list(images.loc[images.problem == 3].file_hash.values)
            if len(duplicate_hashes) == 0:
                dataset_stats.p_3_are_dupes = "none"
            else:
                non_duplicate_hashes = list(
                    images.loc[(images.problem != 3)].file_hash.values
                )
                not_actual_duplicates = list(
                    set(duplicate_hashes) - set(non_duplicate_hashes)
                )
                if len(not_actual_duplicates) == 0:
                    dataset_stats.p_3_are_dupes = "yes"
                else:
                    dataset_stats.p_3_are_dupes = "no, " + str(
                        len(not_actual_duplicates)
                    )

            class_list = list(classes.sort_values(by="class_id").class_name.values)
            dataset_stats.class_alpha = str(
                dataset_utils.is_list_in_alphabetical_order(class_list)
            )

            class_ids = list(classes.sort_values(by="class_id").class_id.values)
            dataset_stats.class_gap = str(
                not all(
                    [
                        ((class_ids[x] - class_ids[x - 1]) == 1)
                        for x in range(1, len(class_ids))
                    ]
                )
            )

            # get the file_hash counts for only problem=0 images
            dupe_hashes = images[images.problem == 0].file_hash.value_counts()
            dataset_stats.p_0_unique_hashes = len(dupe_hashes)
            dataset_stats.p_0_dupes = (
                dataset_stats.p_0 - dataset_stats.p_0_unique_hashes
            )

            (
                dupe_hashes_with_more_than_one_class_id,
                dupe_hashes_image_count,
            ) = dataset_utils.problem_zero_dupes(images, silent=True)

            dataset_stats.p_0_dupes_with_gt_one_class = len(
                dupe_hashes_with_more_than_one_class_id
            )
            dataset_stats.p_0_dupes_with_gt_count = dupe_hashes_image_count

    datasets += [dataset_stats]


datasets_df = pd.DataFrame(data=datasets)
dataset_utils.chdir_with_create(root_directory + "0_collection/")
datasets_df.to_excel("dashboard.xlsx", index=False)
