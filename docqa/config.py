from os import environ
from os.path import join, expanduser, dirname

"""
Global config options
"""

DATA_DIR = environ.get("DATA_DIR", join(expanduser("~"), "data"))

VEC_DIR = join(DATA_DIR, "glove")
SQUAD_SOURCE_DIR = join(DATA_DIR, "squad")
SQUAD_TRAIN = join(SQUAD_SOURCE_DIR, "train-v1.1.json")
SQUAD_DEV = join(SQUAD_SOURCE_DIR, "dev-v1.1.json")


TRIVIA_QA = join(DATA_DIR, "triviaqa")
TRIVIA_QA_UNFILTERED = join(DATA_DIR, "triviaqa-unfiltered")
LM_DIR = join(DATA_DIR, "lm")
DOCUMENT_READER_DB = join(DATA_DIR, "doc-rd", "docs.db")


CORPUS_DIR = environ.get("CORPUS_DIR", join(dirname(dirname(__file__)), "data"))
