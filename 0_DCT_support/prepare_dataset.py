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

# INSERT ANY DATASET-SPECIFIC NOTES HERE
#
# e.g. there were multiple shuffles of train/val/test, so no image belong to any one set
# e.g. a folder was duplicated and images were repeated by mistake
# e.g. non-dataset images (like header image files) exist
#


#
# 1. constants
#

# e.g., url, files, and md5sums
# DATASET_URL_1 = "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
# DATASET_FILE_1 = "freiburg_groceries_dataset.tar.gz"
# DATASET_SUM_1 = "4d7a9d202da5f0d0f09e69eca4c28bf0"


#
# 1. imports
#

### ONLY BASE PYTHON MODULES ARE ALLOWED (except for use of dataset_utils) ###
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

### Feel free the change the body of the functions, but not the function definitions (APIs) ###

### DO NOT CHANGE THE FUNCTION DEFINITION (API) ###
def download_dataset(dataset_root_path: str):
    """This function:
    - downloads the necessary archive and supporting files (if possible)
    - puts them in ./archives/
    """
    print("download dataset")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    ######### INSERT CODE FOR DOWNLOADING NECESSARY DATASET LINKS #########

    # download tar/zip/7z/txt/etc to ./archives/
    # don't download file again
    #
    # if not os.path.isfile(DATASET_FILE_1):
    #    run_cmd_get_output(["wget", DATASET_URL_1])

    #######################################################################


### DO NOT CHANGE THE FUNCTION DEFINITION (API) ###
def verify_md5sums(dataset_root_path: str) -> bool:
    """This function:
    - verifies their md5sums (returns True if all sums match)
    """
    print("verify md5sums")

    # change dir
    chdir_with_create(dataset_root_path + "archives/")

    ######### INSERT CODE FOR VERIFYING MD5SUMs FOR EACH FILE #########
    # verify md5sums for each downloaded file

    # the return should only be True if all md5sums match
    return False

    ###################################################################


### DO NOT CHANGE THE FUNCTION DEFINITION (API) ###
def extract_dataset(dataset_root_path: str):
    """This function:
    - extracts any necessary files from the archives into ./original_files/
    """
    print("extract dataset")

    # change dir
    chdir_with_create(dataset_root_path + "original_files/")

    ######### INSERT CODE FOR EXTRACTING DATASET ARCHIVES #########

    # extract the entire archive as-is into ./original_files/

    # original_file_path = dataset_root_path + "archives/" + DATASET_FILE_1

    # run_cmd_get_output(["tar", "-xf", original_file_path])

    ################################################################


### DO NOT CHANGE THE FUNCTION DEFINITION (API) ###
def convert_dataset(dataset_root_path: str):
    """IF NECESSARY, this function:
    - converts any non-image files (like .mat) to .png files and puts them in ./converted_files/
    """
    print("no conversion needed for dataset")

    ######### INSERT CODE TO CONVERT DATASET (IF NECESSARY) #########

    # change dir
    # chdir_with_create(dataset_root_path + "converted_files/")

    # if there are any conversions (.mat or otherwise), the output .png files will be placed here

    #################################################################


### DO NOT CHANGE THE FUNCTION DEFINITION (API) ###
def parse_dataset(
    dataset_root_path: str,
) -> tuple[list[DatasetImage], dict[str, int]]:
    """This returns a list of all images within the dataset.
    This includes **all images** in both ./original_files/ and, if necessary, ./converted_files/"""

    print("parse dataset")

    # list of images (DatasetImage)
    # class_id can be int or None
    image_list: list[DatasetImage] = []
    # dictionary mapping class labels (text) to label integers
    class_dict: dict[str, int] = {}

    ######### INSERT CODE TO PARSE DATASET #########

    # insert all code here
    # If you want to break out specific functions, insert them below at section 3.

    ################################################

    return (image_list, class_dict)


#
# 3. dataset-specific helper functions
#


# if the file is run directly, it will fully prepare the dataset from scratch
### DO NOT CHANGE ANY CODE IN HERE ###
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

    # this is only required if original files were in .mat or similar
    convert_dataset(dataset_root_path)

    # this creates the sqlite file
    create_images_database(
        dataset_root_path,
        parse_dataset,
    )
