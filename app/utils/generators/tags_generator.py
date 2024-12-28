from app.config.constants import RAW_DATA_TAGS_ARABIC_PATH, RAW_DATA_TAGS_ENGLISH_PATH, RAW_DATA_TAGS_FRENCH_PATH, \
    RAW_DATA_TAGS_PATH, RAW_DATA_TAGS_JSON, RAW_DATA_TAGS_TXT, RAW_DATA_TAGS_XML
from app.utils.raw_tags import POS_TAGS, POS_TAGS_ARABIC, POS_TAGS_ENGLISH, POS_TAGS_FRENCH
import json
from os import path, makedirs


def generate_tags(tag_type=None, save_format="all"):
    """
    Generate tags in txt, xml or json format
    :param tag_type: arabic | english | french | None
    :param save_format: txt | json | xml | all
    :return:
    """
    if save_format == "txt":
        generate_tags_txt(tag_type)
    elif save_format == "json":
        generate_tags_json(tag_type)
    elif save_format == "xml":
        generate_tags_xml(tag_type)
    elif save_format == "all":
        generate_tags_txt(tag_type)
        generate_tags_json(tag_type)
        generate_tags_xml(tag_type)
    else:
        raise ValueError("Invalid save format! Please choose one of the following: txt | json | xml | all")


def generate_tags_xml(tag_type=None):
    """
    Generate tags in xml format
    :param tag_type: arabic | english | french | None
    :return:
    """
    tags, file_path = init_generation_params(tag_type)
    if tags is not None and file_path is not None:
        with open(path.join(file_path, RAW_DATA_TAGS_XML), "w") as file:
            file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            file.write("<tags>\n")
            for idx in range(len(tags)):
                file.write(f"\t<tag id=\"{POS_TAGS[idx]}\">{tags[idx]}</tag>\n")
            file.write("</tags>")
    else:
        raise ValueError("Error occurred while generating tags! Please check the tag type! and also the save file path!")


def generate_tags_json(tag_type=None):
    """
    Generate tags in json format
    :param tag_type: arabic | english | french | None
    :return:
    """
    tags, file_path = init_generation_params(tag_type)
    if tags is not None and file_path is not None:
        # save tags in json format with keys from POS_TAGS  and values from tags
        json_tags = {POS_TAGS[idx]: tags[idx] for idx in range(len(tags))}
        with open(path.join(file_path, RAW_DATA_TAGS_JSON), "w") as file:
            # pretty print the json file and write as is to the file ( no encoding )
            json.dump(json_tags, file, indent=4, ensure_ascii=False)
    else:
        raise ValueError("Error occurred while generating tags! Please check the tag type! and also the save file path!")


def generate_tags_txt(tag_type=None):
    """
    Generate tags in txt format
    :param tag_type: arabic | english | french | None
    :return:
    """
    tags, file_path = init_generation_params(tag_type)
    if tags is not None and file_path is not None:
        with open(path.join(file_path, RAW_DATA_TAGS_TXT), "w") as file:
            idx = 0
            for tag in tags:
                file.write(f"{idx}\t{tag}\n")
                idx += 1
    else:
        raise ValueError("Error occurred while generating tags! Please check the tag type! and also the save file path!")


def init_generation_params(tag_type=None):
    """
    Initialize the generation parameters
    :param tag_type: arabic | english | french | None
    :return:
    """
    tags = None
    file_path = None
    if tag_type == "arabic":
        tags = POS_TAGS_ARABIC
        file_path = RAW_DATA_TAGS_ARABIC_PATH
    elif tag_type == "english":
        tags = POS_TAGS_ENGLISH
        file_path = RAW_DATA_TAGS_ENGLISH_PATH
    elif tag_type == "french":
        tags = POS_TAGS_FRENCH
        file_path = RAW_DATA_TAGS_FRENCH_PATH
    else:
        tags = POS_TAGS
        file_path = RAW_DATA_TAGS_PATH
    return tags, file_path


def mkdirs_save_paths():
    """
    Create directories if they don't exist
    :return:
    """
    if not path.exists(RAW_DATA_TAGS_PATH):
        makedirs(RAW_DATA_TAGS_PATH)
    if not path.exists(RAW_DATA_TAGS_ARABIC_PATH):
        makedirs(RAW_DATA_TAGS_ARABIC_PATH)
    if not path.exists(RAW_DATA_TAGS_ENGLISH_PATH):
        makedirs(RAW_DATA_TAGS_ENGLISH_PATH)
    if not path.exists(RAW_DATA_TAGS_FRENCH_PATH):
        makedirs(RAW_DATA_TAGS_FRENCH_PATH)


def generate_tags_abrv():
    """
    Generate tags with abbreviation
    :return:
    """
    generate_tags()


def generate_tags_arabic():
    """
    Generate tags in arabic
    :return:
    """
    generate_tags(tag_type="arabic")


def generate_tags_english():
    """
    Generate tags in english
    :return:
    """
    generate_tags(tag_type="english")


def generate_tags_french():
    """
    Generate tags in french
    :return:
    """
    generate_tags(tag_type="french")


if __name__ == "__main__":
    mkdirs_save_paths()
    generate_tags_abrv()
    generate_tags_arabic()
    generate_tags_french()
    generate_tags_english()
    print("Tags generated successfully!")
