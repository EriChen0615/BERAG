from vqa_datasets import load_vqa_dataset, load_vqa_passages
import os

if __name__ == '__main__':
    OKVQA_TARGET = "data/OKVQA_with_PseudoGold"
    OKVQA_IMG_BASEDIR = "../vqa_data/KBVQA_data/ok-vqa/"
    for split in ["train", "test", "valid"]:
        okvqa_ds = load_vqa_dataset("OKVQA", split=split, img_basedir=OKVQA_IMG_BASEDIR)
        okvqa_passages = load_passages("OKVQA", split=split)