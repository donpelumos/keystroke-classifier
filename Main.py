from utils.FileUtil import FileUtil


def main():
    try:
        f = FileUtil()
        f.readfile("demo.txt")
    except Exception as e:
        a = 4
        b = 5
    print("Main Method")


main()
