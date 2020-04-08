import sys


class FileUtil:
    """Class used for reading, writing and appending files from the resource directory(folder)"""
    __base_path = sys.path[0]

    def __init__(self):
        pass

    def readfile(self, filename):
        path = self.__base_path + "\\resources\\" + filename
        file = open(path)
        return file.read()