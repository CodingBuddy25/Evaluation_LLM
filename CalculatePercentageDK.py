import re
from matplotlib import pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/977491/comparing-two-txt-files-using-difflib-in-python
# evaluate what is different between the two evaluations

def remove_hl(file: object) -> object:
    # Remove all instances of \hl{...}
    cleaned_text = file.replace("}", "")
    cleaned_text = cleaned_text.replace("\\hl{", "")
    # In the latex document this I have added \> and \&
    cleaned_text = cleaned_text.replace("\\", "")

    return cleaned_text


def remove_characters(file, Menu):
    # Finding the amount of characters that are NOT highlighted
    if Menu == 1:
        removed_characters = re.sub(r'\\hl{.*?}', '', file, flags=re.DOTALL)
    if Menu == 2:
        removed_characters = re.findall(r"\[(https?://[^\]]+)\]", file)
    return removed_characters

def Evaluate_domain_knowledge():
    results = []

    ##Example of files
    files = [
    "audit-O2C-PG",
    "environmental-P2P-Walmart",
    "inefficiencies-AP-GE",
    "IT-IT-Volvo",
    "operational-Travel-Google",
    "regulatory-loan-wellsfargo"
    ]
    #Example of foldernames, you can add multiple foldernames
    foldernames = ["DoS2_DK/DoS2_run3/"]
    for foldername in foldernames:
        for file in files:
            file = foldername + file
            print("file = ",file)
            print("_________________________FILE:", file, "_______________________")
            latex_text_opened = open(file, 'r', encoding="utf-8")
            latex_text_read = latex_text_opened.read()
            cleaned_text = remove_hl(latex_text_read)
            char_count = len(cleaned_text)
            print("The amount of characters in the document is:", char_count)

            char_few_count = len(remove_characters(latex_text_read, Menu))
            percentage = ((char_count - char_few_count) / char_count) * 100
            print("The amount of characters highlighted are:", char_count - char_few_count)
            print("The percentage of highlighted characters therefore is:((", char_count, "-", char_few_count, ")/",
                  char_count, ")*100 =",
                  percentage)
            results.append([file,percentage])
            print("\n")
    print(results)

Menu = 1

Evaluate_domain_knowledge()
