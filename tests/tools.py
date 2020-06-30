import os


def get_path_to_test_data() -> str:
    cur_working_directory = os.getcwd()
    if os.path.split(cur_working_directory)[-1] != "tests":
        cur_working_directory = os.path.join(cur_working_directory, "tests")
    return os.path.join(cur_working_directory, "resources", "java-test")
