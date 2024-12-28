import os

# BASE PATH
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# RAW DATA PATH
ROOT_PATH = os.path.join(BASE_PATH, "..", "..")
DATA_PATH = os.path.join(BASE_PATH, ROOT_PATH, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
RAW_DATA_FILES_PATH = os.path.join(DATA_PATH, 'raw-files')

# TAGS FILES PATHS
RAW_DATA_TAGS_PATH = os.path.join(RAW_DATA_FILES_PATH, 'raw-tags-indexed')
RAW_DATA_TAGS_ARABIC_PATH = os.path.join(RAW_DATA_FILES_PATH, 'raw-tags-arabic-indexed')
RAW_DATA_TAGS_ENGLISH_PATH = os.path.join(RAW_DATA_FILES_PATH, 'raw-tags-english-indexed')
RAW_DATA_TAGS_FRENCH_PATH = os.path.join(RAW_DATA_FILES_PATH, 'raw-tags-french-indexed')

# TAGS FILES NAMES
RAW_DATA_TAGS_TXT = "tags.txt"
RAW_DATA_TAGS_JSON = "tags.json"
RAW_DATA_TAGS_XML = "tags.xml"
