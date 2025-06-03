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

    #use this structure to loop through the files that you want analysed
    files = [
    "audit-O2C-PG",
    "environment-P2P-Walmart",
    "inefficiencies-AP-GE",
    "IT-It-Volvo",
    "operational-travel-google",
    "regulatory-loan-wellsfargo"
    ]
    foldernames = ["DoS2_DK/DoS2_run2/"]
    for foldername in foldernames:
        for file in files:
            file = foldernames + file
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
#
# files = ['/AP_1.txt', '/AP_2.txt', '/IT_cyber_1.txt',
#              '/IT_cyber_2.txt', '/Loan_application_1.txt', '/Loan_application_2.txt',
#              '/O2C_1.txt', '/O2C_2.txt', '/P2P_1.txt', '/P2P_2.txt',
#              '/Travel_expenses_1.txt', '/Travel_expenses_2.txt',]
# graders = ["Evaluate_R","Evaluate_N","Evaluate_M2","Evaluate_M"]

Evaluate_domain_knowledge()
