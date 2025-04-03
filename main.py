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

def Evaluate_domain_knowledge(Menu, graders, files):
    results = []
    for grader in graders:
        for file in files:
            file = grader + file
            print("_________________________FILE:", file, "_______________________")
            latex_text_opened = open(file, 'r')
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
    return results

def preprocessing(files, graders, results):

    return results
def plotting_results(files, graders, results):
    """
        This function uses the previously generated percentages to generate graphs. They are saved.
        A small portion of the results looks like this:
        results = [['Evaluate_R/AP_1.txt', 84.45212240868707], ['Evaluate_R/AP_2.txt', 73.08289789577744], ['Evaluate_R/IT_cyber_1.txt', 85.80359651587524]]
       It is a nested list of the percentages
    """

    fig = plt.figure(figsize=(12, 6))


    #plot per grader line charts
    renske = results[0:12]
    naomi = results[12:24]
    max2 = results[24:36]
    max = results[36:]
    values_R = [item[1] for item in renske]
    values_N = [item[1] for item in naomi]
    values_M = [item[1] for item in max]
    values_M2 = [item[1] for item in max2]
    plt.ylim(0, 100)

    plt.plot(files, values_R, label='Renske')
    plt.plot(files, values_N, label='Naomi')
    plt.plot(files, values_M, label='Max1')
    plt.plot(files, values_M2, label='Max2')

    plt.legend(["Renske","Naomi","Max1","Max2"])
    plt.tight_layout()
    plt.title("Percentage DK per file per grader")
    plt.ylabel('Percentage (%)')
    plt.tight_layout()

    results_per_paper = []
    #Will be a nested list [[56.35,45.64,56.65,45.65],[89.87,67.98,56.65,45.65]...]

    for count in range(12):
        elements_per_paper = [renske[count][1], naomi[count][1], max2[count][1], max[count][1]]
        results_per_paper.append(elements_per_paper)

    fig2 = plt.figure(figsize=(12, 6))
    plt.boxplot(results_per_paper,vert=False)
    plt.title("Box plot of DK per file")
    plt.tight_layout()
    plt.xlim(0, 100)


    fig3 = plt.figure(figsize=(10, 6))
    averages = [np.average(paper_stats) for paper_stats in results_per_paper]
    sds = [np.std(paper_stats) for paper_stats in results_per_paper]
    # plt.bar(files, averages)
    plt.bar(["AP_1","AP_2","INC_1","INC_2","LA_1","LA_2", "O2C_1", "O2C_2", "P2P_1", "P2P_2", "EXP_1", "EXP_2"], averages)
    # plt.errorbar(files, averages, yerr=sds, fmt="o", color="r")
    plt.errorbar(["AP_1","AP_2","INC_1","INC_2","LA_1","LA_2", "O2C_1", "O2C_2", "P2P_1", "P2P_2", "EXP_1", "EXP_2"], averages, yerr=sds, fmt="o", color="r")
    plt.xticks(rotation='vertical')
    # plt.title("Box plot with standard deviations of domain knowledge averages")
    plt.ylabel('Percentage (%)')
    plt.show()

Menu = 1

files = ['/AP_1.txt', '/AP_2.txt', '/IT_cyber_1.txt',
             '/IT_cyber_2.txt', '/Loan_application_1.txt', '/Loan_application_2.txt',
             '/O2C_1.txt', '/O2C_2.txt', '/P2P_1.txt', '/P2P_2.txt',
             '/Travel_expenses_1.txt', '/Travel_expenses_2.txt',]
graders = ["Evaluate_R","Evaluate_N","Evaluate_M2","Evaluate_M"]

while Menu != 0:
    Menu = int(input("1 for evaluate domain knowledge, 0 to quit:      "))
    if Menu == 1:
        results = Evaluate_domain_knowledge(Menu, graders, files)
        plotting_results(files, graders, results)
