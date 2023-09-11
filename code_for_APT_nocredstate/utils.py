import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args
