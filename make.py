import glob as g
import os
import sys

def data():
    paths = g.glob("code/transform_*")

    for path in paths:
        print(f"Running: {path}")
        os.system("python "+path)

def clean():
    paths = g.glob("data/*/*.npy")
    for path in paths:
        os.remove(path)

def model():
    print("Choose model (type model name):")
    models = ["gbd1", "ube4b"]
    [print(model) for model in models]
    user_input = input()

    check_pass = False
    for model in models:
        if user_input == model:
            check_pass = True
    
    if check_pass != True:
        print("Model does not exist")
        return

    model_paths = g.glob("code/model_*.py")
    for path in model_paths:
        if user_input in path:
            os.system("python " + path)

if __name__ == "__main__":
    make_list = ["data", "clean", "model"]
    args = sys.argv
    
    if len(args) == 1:
        print("Run with one or more of following args:")
        for make in make_list:
            print("    " + make)

    for arg in args:
        for make in make_list:
            if arg == make:
                locals()[make]()