import re  # regular expression
import sys

import openai
import os
from langchain_openai import ChatOpenAI
import pm4py
import numpy as np
from matplotlib import pyplot as plt

def LLM_domain_knowledge(file, specific_file_question):
    """This function is there that when the different files need to be analysed, there are all slightly different
     evaluation prompts. In creating_the_question we will look at the file (input) and then return the right question.
    #PM-LLM-benchmark question was
    # Given the following question: ... How would you grade the following answer from 1 (minimum) to 10 (maximum)?
    #I have changed this question to: see evaluation_question"""

    file_path = os.getcwd()
    file_path = os.path.join(file_path, "DoS0_DK", "Evaluate_M",f"{file}")
    print(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        answer_text = file.read()

    evaluation_question = f"""Given the following question: {specific_file_question}"
        And the answer to the question:
        {answer_text}
        Please evaluate the above answer on the amount of specific external ecosystem domain knowledge. 
        It is specific external ecosystem domain knowledge when it is linked to the company and the step being analyzed 
        within the company and process mining domain knowledge. 
        Return a percentage of the text that has specific external ecosystem domain knowledge.
        """
    return evaluation_question

def finding_percentage_dk(answer):
    percentage_or_none = re.search(r'(\d+)%', answer) #\d+ means that it is looking for one or more characters before the % sign.
    if percentage_or_none:
        percentage = int(percentage_or_none.group(1))
        #    (\d+) → This is Group 1, which captures one or more digits (\d+).
        #    group(0) returns the entire match (i.e., "80%").
        #    group(1) returns only the first capturing group, which is "80".
        return percentage
    else:
        return None

def multiple_run_dkanalysis(dict_with_percentages):
    analysis = {}
    raw_percentages = []
    for file, values in dict_with_percentages.items():
        list_no_Nones = [percentage for percentage in values if percentage is not None]  # Filter out None values
        raw_percentages.append(list_no_Nones)
        analysis[file] = {
            'Min': [],
            'Max': [],
            'Q1': [],
            'Q3': [],
            'Median': [],
            'sd': [],
            'Average': []}
        analysis[file]['Min'] = np.min(list_no_Nones)
        analysis[file]['Max'] = np.max(list_no_Nones)
        analysis[file]['Q1'] = np.quantile(list_no_Nones,0.25)
        analysis[file]['Q3'] = np.quantile(list_no_Nones,0.75)
        analysis[file]['Median'] = np.median(list_no_Nones)
        analysis[file]['sd'] = np.std(list_no_Nones)
        analysis[file]['Average'] = np.average(list_no_Nones)

    print("raw percentages: ", raw_percentages)
    print(analysis)
    return(analysis,raw_percentages)


def LLM_prompt_references(references):
    """This function is there that when the different files need to be analysed, there are all slightly different
    evaluation prompts. In creating_the_question we will look at the file (input) and then return the right question.
    PM-LLM-benchmark question was
    Given the following question: ... How would you grade the following answer from 1 (minimum) to 10 (maximum)?
    I have changed this question to:"""

    print(len(references), "length references")


    evaluation_question = f"""Given the following references: {references}
    Please evaluate the references and put them into seven categories: self-published company papers, academic publications, non-academic research, non-academic other source, social media or press release, link does not exist (anymore) and law documents.
        The explanation per category is as follows:
        - Self-published company papers - this means that the source is from a page of the company that is getting analyzed by the PoC. For example, www.media.volvocars.com for the Volvo prompt (IT_cyber_1 and IT_cyber_2) and www.pgsupplier.com the P&G prompt (O2C_1 and O2C_2).
        - Academic publications - these references link to an acknowledged publications platform such as www.tandfonline.com, www.nature.com and link.springer.com.
        - Non-academic research - references that lead to research done by companies. Often these are large consulting companies that have done a large market research. To qualify as a ``research" the paper needs to have links to other research or be done by multiple researchers. It cannot be for example a blog of one person who has done a small internet research. This is a fine line, yet this definition should really encapsulate research that has been done with significant effort, significant evidence or referencing and/or a significant amount of contributors. 
        - Non-academic source (not research, often a blog explaining terminology or a process). Any source that is non-academic and that does not fall in the category above. Often these are blogs of people that have done research into the definition or process in a particular domain. 
        - Social media or press release. This includes articles and news that is published on social media platforms or press platforms. References from social media platforms are often from LinkedIn and press release platforms range from papers such as ``The Guardian" to ``Forbes". 
        - Link does not exist (anymore) - this is a link that does not lead to an article. It may be that the link once existed and that the company has removed the article or it may be that it has never existed at all.
        - Law documents - this is a small category made for law documents. These are official documents that factually state the laws. The domain of loan applications for the company Wells Fargo.    
                
        The references are elements in a list.  
        TASK: Give the frequency, in a number per category. You do not need to provide the links. It is very important that you check ALL the references! Do not skip any. If a reference cannot be classified, classify it as Non-academic other source."""
    return evaluation_question

def get_statistics_human():
    results = [['Evaluate_R/AP_1.txt', 84.45212240868707], ['Evaluate_R/AP_2.txt', 73.08289789577744], ['Evaluate_R/IT_cyber_1.txt', 85.80359651587524], ['Evaluate_R/IT_cyber_2.txt', 93.39330259278857], ['Evaluate_R/Loan_application_1.txt', 72.65187177762833], ['Evaluate_R/Loan_application_2.txt', 65.93052852223387], ['Evaluate_R/O2C_1.txt', 78.41296928327644], ['Evaluate_R/O2C_2.txt', 72.86648825453464], ['Evaluate_R/P2P_1.txt', 82.58913695434855], ['Evaluate_R/P2P_2.txt', 96.98161839517059], ['Evaluate_R/Travel_expenses_1.txt', 78.8384177003017], ['Evaluate_R/Travel_expenses_2.txt', 89.5984738372093], ['Evaluate_N/AP_1.txt', 72.12548466690166], ['Evaluate_N/AP_2.txt', 62.56620295176894], ['Evaluate_N/IT_cyber_1.txt', 41.10127826941986], ['Evaluate_N/IT_cyber_2.txt', 57.86173026067246], ['Evaluate_N/Loan_application_1.txt', 55.69667538289129], ['Evaluate_N/Loan_application_2.txt', 61.64670658682635], ['Evaluate_N/O2C_1.txt', 67.71867245657567], ['Evaluate_N/O2C_2.txt', 61.66195926861899], ['Evaluate_N/P2P_1.txt', 68.65062848580705], ['Evaluate_N/P2P_2.txt', 71.06333921063339], ['Evaluate_N/Travel_expenses_1.txt', 75.85889056477292], ['Evaluate_N/Travel_expenses_2.txt', 73.64214350590372], ['Evaluate_M2/AP_1.txt', 79.63341557983784], ['Evaluate_M2/AP_2.txt', 80.05647723261559], ['Evaluate_M2/IT_cyber_1.txt', 87.23747980613894], ['Evaluate_M2/IT_cyber_2.txt', 89.2365835222978], ['Evaluate_M2/Loan_application_1.txt', 81.62594336098034], ['Evaluate_M2/Loan_application_2.txt', 73.83429384028142], ['Evaluate_M2/O2C_1.txt', 68.4528887165568], ['Evaluate_M2/O2C_2.txt', 77.69436598781031], ['Evaluate_M2/P2P_1.txt', 74.35043304463692], ['Evaluate_M2/P2P_2.txt', 78.25703628348593], ['Evaluate_M2/Travel_expenses_1.txt', 74.19138595609184], ['Evaluate_M2/Travel_expenses_2.txt', 59.48228882833787], ['Evaluate_M/AP_1.txt', 84.84698914116485], ['Evaluate_M/AP_2.txt', 63.33145036011862], ['Evaluate_M/IT_cyber_1.txt', 83.62959319890395], ['Evaluate_M/IT_cyber_2.txt', 81.63064833005895], ['Evaluate_M/Loan_application_1.txt', 57.863700828543706], ['Evaluate_M/Loan_application_2.txt', 78.20915619389586], ['Evaluate_M/O2C_1.txt', 88.12103389568179], ['Evaluate_M/O2C_2.txt', 90.13133486681012], ['Evaluate_M/P2P_1.txt', 89.13948069241012], ['Evaluate_M/P2P_2.txt', 84.95695207104603], ['Evaluate_M/Travel_expenses_1.txt', 56.47295260425389], ['Evaluate_M/Travel_expenses_2.txt', 80.42116728692021]]
    #These results are generated by the main.py programme.


    #plot per grader line charts
    renske = results[0:12]
    naomi = results[12:24]
    max2 = results[24:36]
    max = results[36:]


    results_per_paper = []
    #Will be a nested list [[56.35,45.64,56.65,45.65],[89.87,67.98,56.65,45.65]...]

    for count in range(12):
        elements_per_paper = [renske[count][1], naomi[count][1], max2[count][1], max[count][1]]
        results_per_paper.append(elements_per_paper)

    averages = [np.average(paper_stats) for paper_stats in results_per_paper]
    sds = [np.std(paper_stats) for paper_stats in results_per_paper]

    return averages,sds


def plotting_results(raw_percentages, evaluation_statistics):
    """
        This function uses the previously generated statistics to generate graphs. They are saved.
        raw_percentages = [[60, 70, 60], [80, 70, 46], [60, 70]]
       A nested list of the raw percentages
       evaluation_statistics = {'AP_1.txt': {'Min': 60, 'Max': 70, 'Q1': 60.0, 'Q3': 65.0, 'Median': 60.0, 'sd': 4.714045207910317, 'Average': 63.333333333333336}, 'AP_2.txt': {'Min': 46, 'Max': 80, 'Q1': 58.0, 'Q3': 75.0, 'Median': 70.0, 'sd': 14.2672897060218, 'Average': 65.33333333333333}, 'IT_cyber_1.txt': {'Min': 60, 'Max': 70, 'Q1': 62.5, 'Q3': 67.5, 'Median': 65.0, 'sd': 5.0, 'Average': 65.0}}
        A nested dictionary in a dictionary. Contains the results of the statistical analysis.
    """
    Menu = 10

    while Menu != 0:
        Menu = int(input(
            "1 LLM analysis, 2 LLM and human analysis, 0 to quit:      "))
        if Menu == 1:
            fig = plt.figure(figsize=(12, 6))
            plt.boxplot(raw_percentages,vert=False)
            list_of_number_of_files = [text_name + 1 for text_name in range(len(evaluation_statistics.keys()))]
            #Looks at the keys. If there are three keys, it will return [1,2,3], if there are 4: [1,2,3,4] etc.
            print(list_of_number_of_files, list(evaluation_statistics.keys()))
            plt.yticks(list_of_number_of_files, list(evaluation_statistics.keys()))
            plt.tight_layout()

            fig2 = plt.figure(figsize=(10, 6))
            averages = [statistics_per_file['Average'] for statistics_per_file in evaluation_statistics.values()]
            sd_list = [statistics_per_file['sd'] for statistics_per_file in evaluation_statistics.values()]
            # plt.bar(list(evaluation_statistics.keys()), averages)
            plt.bar(["P2P_1","P2P_2","O2C_1","O2C_2","INC_1","INC_2", "AP_1", "AP_2", "EXP_1", "EXP_2", "LA_1", "LA_2"], averages)
            # plt.errorbar(list(evaluation_statistics.keys()), averages, yerr=sd_list, fmt="o", color="r")
            plt.errorbar(["P2P_1","P2P_2","O2C_1","O2C_2","INC_1","INC_2", "AP_1", "AP_2", "EXP_1", "EXP_2", "LA_1", "LA_2"], averages, yerr=sd_list, fmt="o", color="r")

            plt.xticks(rotation='vertical')
            # plt.title("Percentage of external domain knowledge (using the method LLM-as-a-judge) per output file")
            # Had to remove the title
            plt.ylabel('Percentage (%)')
            plt.tight_layout()
            plt.show()

        if Menu == 2:
            fig = plt.figure(figsize=(12, 6))
            plt.boxplot(raw_percentages, vert=False)
            list_of_number_of_files = [text_name + 1 for text_name in range(len(evaluation_statistics.keys()))]
            # Looks at the keys. If there are three keys, it will return [1,2,3], if there are 4: [1,2,3,4] etc.
            print(list_of_number_of_files, list(evaluation_statistics.keys()))
            plt.yticks(list_of_number_of_files, list(evaluation_statistics.keys()))
            plt.tight_layout()

            fig2 = plt.figure(figsize=(10, 6))
            averages_LLM = [statistics_per_file['Average'] for statistics_per_file in evaluation_statistics.values()]
            sd_list_LLM = [statistics_per_file['sd'] for statistics_per_file in evaluation_statistics.values()]
            averages_human,sd_list_human  = get_statistics_human()
            x = np.arange(12)

            plt.bar(x - 0.2, averages_human, width=0.4, color='cyan')
            plt.bar(x + 0.2, averages_LLM, width=0.4, color='orange')

            plt.errorbar(x - 0.2, averages_human, yerr=sd_list_human, fmt="o", color="r")
            plt.errorbar(x + 0.2, averages_LLM, yerr=sd_list_LLM, fmt="o", color="r")

            plt.legend(["human", "LLM"])
            plt.xticks(x,
                       ["P2P_1", "P2P_2", "O2C_1", "O2C_2", "INC_1", "INC_2", "AP_1", "AP_2", "EXP_1", "EXP_2", "LA_1",
                        "LA_2"], rotation='vertical')
            # plt.title("Percentage of external domain knowledge (using the method LLM-as-a-judge) per output file")
            # Had to remove the title
            plt.ylabel('Percentage (%)')
            plt.tight_layout()
            plt.show()
def perform_DK_LLM_evaluation():
    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    api_key = os.getenv('OPENAI_API_KEY')
    model = "gpt-3.5-turbo"

    evaluation_texts = {
    'AP_1.txt': [],
    'AP_2.txt': [],
    'IT_cyber_1.txt': [],
    'IT_cyber_2.txt': [],
    'Loan_application_1.txt': [],
    'Loan_application_2.txt': [],
    'O2C_1.txt': [],
    'O2C_2.txt': [],
    'P2P_1.txt': [],
    'P2P_2.txt': [],
    'Travel_expenses_1.txt': [],
    'Travel_expenses_2.txt': []
}

    #Example answer of the filled dictionary
    # evaluation_texts= {
    #    'P2P_1.txt': [70, 90, 90, 70, 90, 90, 80, 90, 80, 80],
    #     'P2P_2.txt': [80, 70, 60, 60, 0, 70, 80, 0, 70, 80],
    #     'O2C_1.txt': [90, 70, 70, 80, 70, 70, 70, 70, 70, 80],
    #     'O2C_2.txt': [80, 60, 70, 80, 70, 80, 70, 80, 70, 70],
    #     "IT_cyber_1.txt": [70, 70, 80, 70, 60, 70, 70, 70, 70, 80],
    #     'IT_cyber_2.txt': [60, 80, 70, 90, 60, 70, 70, 65, 80, 80],
    #     'AP_1.txt': [70, 90, 80, 70, 80, 80, 60, 60, 80, 70],
    #     'AP_2.txt': [70, 60, 70, 50, 70, 65, 80, 80, 60, 80],
    #     "Travel_expenses_1.txt": [60, 0, 70, 70, 80, 40, 70, 90, 60, 80],
    #     'Travel_expenses_2.txt': [70, 70, 50, 40, 70, 70, 80, 60, 82,85],
    #     'Loan_application_1.txt': [44, 60, 70, 60, 70, 75, 70, 70, 80, 70],
    #     'Loan_application_2.txt': [60, 90, 70, 80, 70, 60, 80, 70, 80, 80]
    # }

    specific_file_question = {
    'AP_1.txt': "Can you find the bottlenecks and inefficiencies in an accounts payable (AP) process at General Electric (GE). What are potential causes for the inefficiencies that you identified?",
    'AP_2.txt': "Can you find the bottlenecks and inefficiencies in an accounts payable (AP) process at General Electric (GE). What are potential causes for the inefficiencies that you identified?",
    'IT_cyber_1.txt': "Find the bottlenecks and inefficiencies in the process of handling IT incidents at Volvo. Provide common causes and remediations for the inefficiencies in this process.",
    'IT_cyber_2 .txt': "Find the bottlenecks and inefficiencies in the process of handling IT incidents at Volvo. Provide common causes and remediations for the inefficiencies in this process.",
    'Loan_application_1.txt': "Can you find the regulatory risks in the loan application process at Wells Fargo bank. What are the regulatory risks for specific steps that you identified?",
    'Loan_application_2.txt': "Can you find the regulatory risks in the loan application process at Wells Fargo bank. What are the regulatory risks for specific steps that you identified?",
    'O2C_1.txt': "Can you find the causes for audit risks in the order to cash process at Procter & Gamble (P&G). What are the audit risks for specific steps that you identified?",
    'O2C_2.txt': "Can you find the causes for audit risks in the order to cash process at Procter & Gamble (P&G). What are the audit risks for specific steps that you identified?",
    'P2P_1.txt': "Can you find the environmental and sustainability risks in the purchase to pay process at Walmart. What are the environmental and sustainability risks for specific steps that you identified?",
    'P2P_2.txt': "Can you find the environmental and sustainability risks in the purchase to pay process at Walmart. What are the environmental and sustainability risks for specific steps that you identified?",
    'Travel_expenses_1.txt': "Can you find the operational risks in the travel expense submission process at Google. What are the operational risks for specific steps that you identified?",
    'Travel_expenses_2.txt': "Can you find the operational risks in the travel expense submission process at Google. What are the operational risks for specific steps that you identified?",
}
    for count in range(1):
        #Change the count in range to 10 when performing the domain knowledge one.
        for file in evaluation_texts.keys():
            print("_________________________FILE:", file, "_______________________")
            prompt_dk = LLM_domain_knowledge(file, specific_file_question[file])
            resp_dk = pm4py.llm.openai_query(prompt_dk, api_key=api_key, openai_model=model)
            # print("Response domain knowledge: \n", resp_dk)
            percentage = finding_percentage_dk(resp_dk)
            evaluation_texts[file].append(percentage)  # Adds the integer or "none" to the list for the file

    results_analysis, raw_percentages = multiple_run_dkanalysis(evaluation_texts)
    plotting_results(raw_percentages, results_analysis)

def preprocess_references(textfile_name):
    files = ['AP_1.txt', 'AP_2.txt', 'IT_cyber_1.txt',
             'IT_cyber_2.txt', 'Loan_application_1.txt', 'Loan_application_2.txt',
             'O2C_1.txt', 'O2C_2.txt', 'P2P_1.txt', 'P2P_2.txt',
             'Travel_expenses_1.txt', 'Travel_expenses_2.txt']
    file_classified = "References/"+textfile_name
    text_opened = open(file_classified, 'r',encoding='utf-8')
    text_read = [line.strip().split(',') for line in text_opened.readlines()]
    text_read = [[item.strip('[]"') for item in line] for line in text_read]
    #Gives this: [['AP_1.txt', 'red', 'https://www.appvizer.com/magazine/accounting-finance/payment-processing/improve-accounts-payable-process'], ['AP_1.txt', 'red', 'https://www.invensis.net/blog/latest-accounts-payable-trends'], ['AP_1.txt', 'red', 'https://nanonets.com/blog/automated-invoice-processing/']]

    text_opened.close()
    final_data_human = []
    final_data_LLM = []
    metadata = []
    colours_to_classification = {'orange': 'CP','green':'SP', 'blue':'Non-AR', 'red': 'Non-AA',  'purple':'SM or PR','grey': 'Law', 'black':'None'}

    for paper_name in files:
        specific_paper_data = [row for row in text_read if row[0] == paper_name]
        # Gives this: [['AP_1.txt', 'red', 'https://www.appvizer.com/magazine/accounting-finance/payment-processing/improve-accounts-payable-process'], ['AP_1.txt', 'red', 'https://www.invensis.net/blog/latest-accounts-payable-trends'], ['AP_1.txt', 'red', 'https://nanonets.com/blog/automated-invoice-processing/']]
        #or in the case of the references the same but then without the colours
        if textfile_name == "All_references_classified.txt":
            specific_paper_dict = {'CP':0, 'SP':0, 'Non-AR':0, 'Non-AA':0, 'SM or PR':0, 'None':0, 'Law':0}
            for row in specific_paper_data:
                classification = row[1]
                specific_paper_dict[colours_to_classification[classification]] += 1
                #The line above takes the colour and uses the dictionary to translate it to the right heading

            metadata.append(len(specific_paper_data))
            metadata.append(paper_name)

            final_data_human.append(specific_paper_dict)
        else:
            final_data_LLM.append(specific_paper_data)

    if textfile_name == "All_references_classified.txt":
        final_data_human.append(metadata)
        #final data human is a list with a dictionary and another list of metadata right after that
        #[{'CP': 2, 'SP': 0, 'Non-AR': 8, 'Non-AA': 52, 'SM or PR': 5, 'None': 2, 'Law': 0}, {'CP': 1, 'SP': 3, 'Non-AR': 6, 'Non-AA': 31, 'SM or PR': 4, 'None': 24, 'Law': 0}, ... , [69, 'AP_1.txt', 69, 'AP_2.txt', 47, 'IT_cyber_1.txt', 58, 'IT_cyber_2.txt', 57, 'Loan_application_1.txt', 74, 'Loan_application_2.txt', 65, 'O2C_1.txt', 53, 'O2C_2.txt', 66, 'P2P_1.txt', 65, 'P2P_2.txt', 74, 'Travel_expenses_1.txt', 69, 'Travel_expenses_2.txt']]

    if textfile_name == "References_LLM.txt":
        return final_data_LLM
    else:
        return final_data_human


def perform_references_LLM_evaluation(preprocessed_data):
    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    api_key = os.getenv('OPENAI_API_KEY')
    model = "gpt-4-1106-preview"

    # for count in range(len(preprocessed_data)):
    for count in range(1):

        filename = preprocessed_data[5][0][0]
        part1 = preprocessed_data[5][:20]
        references_p1 = [list[-1] for list in part1]
        part2 = preprocessed_data[5][20:40]
        references_p2 = [list[-1] for list in part2]
        part3 = preprocessed_data[5][40:]
        references_p3 = [list[-1] for list in part3]

        print("_________________________FILE:", filename, "_______________________")
        for element in [references_p1,references_p2,references_p3]:
            prompt_ref = LLM_prompt_references(element)
            print(prompt_ref)
            resp_ref = pm4py.llm.openai_query(prompt_ref, api_key=api_key, openai_model=model)
            print("Response references: \n", resp_ref, "\n")

def retreive_LLM_results():
    """Because the output of the LLM changes per round, in this variable are some of the results stored.
    The LLM could not return a correct format so I will do that manually."""

    #Dictionary made which contains the manually gathered data
    manually_processed_data = [{'CP': 4, 'SP': 3, 'Non-AR': 9, 'Non-AA': 25, 'SM or PR': 5, 'None': 2, 'Law': 0},
                               {'CP': 16, 'SP': 8, 'Non-AR': 12, 'Non-AA': 25, 'SM or PR': 0, 'None': 0, 'Law': 2},
                               {'CP': 10, 'SP': 6, 'Non-AR': 7, 'Non-AA': 14, 'SM or PR': 1, 'None': 0, 'Law': 2},
                               {'CP': 3, 'SP': 6, 'Non-AR': 8, 'Non-AA': 36, 'SM or PR': 2, 'None': 0, 'Law': 0},
                               {'CP': 15, 'SP': 2, 'Non-AR': 12, 'Non-AA': 19, 'SM or PR': 2, 'None': 0, 'Law': 2},
                               {'CP': 7, 'SP': 6, 'Non-AR': 11, 'Non-AA': 38, 'SM or PR': 7, 'None': 0, 'Law': 2},
                               {'CP': 11, 'SP': 1, 'Non-AR': 13, 'Non-AA': 28, 'SM or PR': 4, 'None': 0, 'Law': 0},
                               {'CP': 12, 'SP': 2, 'Non-AR': 11, 'Non-AA': 16, 'SM or PR': 8, 'None': 1, 'Law': 0},
                               {'CP': 20, 'SP': 7, 'Non-AR': 11, 'Non-AA': 20, 'SM or PR': 6, 'None': 0, 'Law': 0},
                               {'CP': 13, 'SP': 9, 'Non-AR': 11, 'Non-AA': 29, 'SM or PR': 2, 'None': 0, 'Law': 0},
                               {'CP': 28, 'SP': 3, 'Non-AR': 7, 'Non-AA': 48, 'SM or PR': 0, 'None': 0, 'Law': 1},
                               {'CP': 32, 'SP': 0, 'Non-AR': 4, 'Non-AA': 26, 'SM or PR': 2, 'None': 0, 'Law': 2},
                               [45, 'AP_1.txt', 63, 'AP_2.txt', 40, 'IT_cyber_1.txt', 55, 'IT_cyber_2.txt',
                                52, 'Loan_application_1.txt', 71, 'Loan_application_2.txt', 57, 'O2C_1.txt',
                                50, 'O2C_2.txt', 64, 'P2P_1.txt', 64, 'P2P_2.txt', 87, 'Travel_expenses_1.txt',
                                66, 'Travel_expenses_2.txt']]

    # manually_processed_data = [{'CP': 0, 'SP': 0, 'Non-AR': 2, 'Non-AA': 10, 'SM or PR': 2, 'None': 7, 'Law': 0},
    #                            {'CP': 0, 'SP': 9, 'Non-AR': 0, 'Non-AA': 0, 'SM or PR': 0, 'None': 1, 'Law': 0},
    #                            {'CP': 9, 'SP': 0, 'Non-AR': 0, 'Non-AA': 0, 'SM or PR': 0, 'None': 9, 'Law': 2},
    #                            {'CP': 5, 'SP': 2, 'Non-AR': 0, 'Non-AA': 8, 'SM or PR': 4, 'None': 0, 'Law': 3},
    #                            {'CP': 0, 'SP': 2, 'Non-AR': 0, 'Non-AA': 9, 'SM or PR': 1, 'None': 1, 'Law': 0},
    #                            {'CP': 0, 'SP': 0, 'Non-AR': 0, 'Non-AA': 4, 'SM or PR': 0, 'None': 2, 'Law': 0},
    #                            [21, 'AP.txt', 10, 'IT_cyber.txt',
    #                             20, 'Loan_application.txt', 22, 'O2C.txt',
    #                             13, 'P2P.txt', 6, 'Travel_expenses.txt']]

    # manually_processed_data = [{'CP': 4, 'SP': 3, 'Non-AR': 9, 'Non-AA': 25, 'SM or PR': 5, 'None': 2, 'Law': 0},
    #                            {'CP': 16, 'SP': 8, 'Non-AR': 12, 'Non-AA': 25, 'SM or PR': 0, 'None': 0, 'Law': 2},
    #                            {'CP': 10, 'SP': 6, 'Non-AR': 7, 'Non-AA': 14, 'SM or PR': 1, 'None': 0, 'Law': 2},
    #                            {'CP': 3, 'SP': 6, 'Non-AR': 8, 'Non-AA': 36, 'SM or PR': 2, 'None': 0, 'Law': 0},
    #                            {'CP': 15, 'SP': 2, 'Non-AR': 12, 'Non-AA': 19, 'SM or PR': 2, 'None': 0, 'Law': 2},
    #                            {'CP': 7, 'SP': 6, 'Non-AR': 11, 'Non-AA': 38, 'SM or PR': 7, 'None': 0, 'Law': 2},
    #                            [45, 'AP_1.txt', 40, 'IT_cyber_1.txt',
    #                             52, 'Loan_application_1.txt', 71, 'O2C_1.txt',
    #                             50, 'P2P_1.txt', 64, 'Travel_expenses_1.txt']]
    return manually_processed_data
def plotting_graphs(final_data):
    Menu = 10
    reserve_final_data = final_data
    while Menu != 0:
        final_data = reserve_final_data
        Menu = int(input("1: total references per paper, 2: category count per paper, 3: Papers in the category of company pages, 4: total references per category, 5: subplots of 3. ALL 7 categories, 6: per file show categories, 0 to quit:      "))
        if Menu == 1:
            # Graph of the reference counts vs the name of the paper
            y_values = final_data[-1][0::2]
            x_labels = final_data[-1][1::2]
            # [1] takes the list with the neccesary data
            # [1::2] takes the first element and does skips of two (aka miss one out)

            plt.figure(figsize=(10, 5))
            plt.bar(x_labels, y_values)

            plt.xlabel('Finenames')
            plt.ylabel('Total amount of references')
            plt.title('Amount of references per paper')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        if Menu == 2:
            #https: // www.geeksforgeeks.org / create - a - grouped - bar - plot - in -matplotlib /

            final_data=final_data[9:]
            #Making a smaller plot with the data of only three papers
            print(len(final_data))

            x_labels = list(final_data[0].keys())
            combined_y_labels = [list(final_data[count].values()) for count in range((len(final_data)-1))]
            # -1 because the final element in the list is the metadata
            print(combined_y_labels)
            print(x_labels)
            print(combined_y_labels[0])
            x = np.arange(7)

            # # Create the bar chart
            plt.figure(figsize=(8, 5))
            plt.bar(x-0.2, combined_y_labels[0], width = 0.2,color='cyan')
            plt.bar(x, combined_y_labels[1],width = 0.2,color='orange')
            plt.bar(x+0.2, combined_y_labels[2],width = 0.2,color='blue')
            plt.legend(final_data[-1][1::2])
            plt.xticks(x,x_labels)

            # Labels and title
            plt.xlabel('Categories')
            plt.ylabel('References count')
            plt.title('References per category')
            plt.xticks(rotation=30)
            plt.tight_layout()

            plt.show()

        if Menu == 3:
            # https: // www.geeksforgeeks.org / create - a - grouped - bar - plot - in -matplotlib /
            #Only 'CP' papers

            combined_y_labels = [list(final_data[count].values()) for count in range((len(final_data) - 1))]
            y_label_only_cp = [list[0] for list in combined_y_labels]

            plt.figure(figsize=(8, 5))
            plt.bar(final_data[-1][1::2], y_label_only_cp, color='orange')

            plt.xlabel('Paper')
            plt.ylabel('References count')
            plt.title('Company paper references')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()





        if Menu ==4:
            for count in range((len(final_data)-1)):
                x_labels = list(final_data[count].keys())
                y_values = list(final_data[count].values())

                # # Create the bar chart
                plt.figure(figsize=(8, 5))
                plt.bar(x_labels, y_values)

                # Labels and title
                plt.xlabel('Categories')
                plt.ylabel('References count')
                plt.title('References per category')
                plt.xticks(rotation=30)
                plt.tight_layout()

                plt.show()

        if Menu ==5:
            #https://stackoverflow.com/questions/45769005/what-does-numpy-ndarray-object-has-no-attribute-barh-mean-and-how-to-rectify
            #https://stackoverflow.com/questions/44598708/understanding-matplotlib-subplots-python
            combined_y_labels = [list(final_data[count].values()) for count in range((len(final_data) - 1))]
            y_label_only_cp = [list[0] for list in combined_y_labels]
            y_label_only_sp = [list[1] for list in combined_y_labels]
            y_label_only_nonar = [list[2] for list in combined_y_labels]
            y_label_only_nonaa = [list[3] for list in combined_y_labels]
            y_label_only_sm = [list[4] for list in combined_y_labels]
            y_label_only_none = [list[5] for list in combined_y_labels]
            y_label_only_law = [list[6] for list in combined_y_labels]

            fig, axs = plt.subplots(4, 2, figsize = (25,15),squeeze=False, constrained_layout=True)
            #Constrained layout: https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
            #squeeze: coordintates of barcharts
            print(final_data[-1][1::2])
            papers_short_name = ['AP1','AP2','INC1','INC2','LOAN1','LOAN2','O2C1','O2C2','P2P1','P2P2','EXP1','EXP2']

            axs[0,0].bar(papers_short_name, y_label_only_cp)
            axs[0,0].set_title('Company papers')
            axs[0,0].set_ylabel('Count')

            axs[0,1].bar(papers_short_name, y_label_only_sp)
            axs[0,1].set_title('Scientific papers')
            axs[0,1].set_ylabel('Count')

            axs[1,0].bar(papers_short_name, y_label_only_nonar)
            axs[1,0].set_title('Non-acad. research')
            axs[1,0].set_ylabel('Count')

            axs[1,1].bar(papers_short_name, y_label_only_nonaa)
            axs[1,1].set_title('Non-acad. articles')
            axs[1,1].set_ylabel('Count')

            axs[2,0].bar(papers_short_name, y_label_only_sm)
            axs[2,0].set_title('Press/Social media')
            axs[2,0].set_ylabel('Count')

            axs[2,1].bar(papers_short_name, y_label_only_none)
            axs[2,1].set_title('Not working link')
            axs[2,1].set_ylabel('Count')

            axs[3,0].bar(papers_short_name, y_label_only_law)
            axs[3,0].set_title('Law documents')
            axs[3,0].set_ylabel('Count')

            axs[3,1].bar(papers_short_name, final_data[-1][0::2])
            axs[3,1].set_title('Total references')
            axs[3,1].set_ylabel('Count')

            plt.show()

        if Menu ==6:
            #Based off 5

            fig, axs = plt.subplots(4, 3, figsize = (24,21),squeeze=False, constrained_layout=True)

            paper_names = final_data[-1][1::2]
            categories_short_name = ['CP','SP','N-AR','N-AA','SMPR','None','Law']

            axs[0,0].bar(categories_short_name,  list(final_data[0].values()))
            axs[0,0].set_title(paper_names[0])
            axs[0,0].set_ylabel('Count')

            axs[0,1].bar(categories_short_name,  list(final_data[1].values()))
            axs[0,1].set_title(paper_names[1])
            axs[0,1].set_ylabel('Count')

            axs[0,2].bar(categories_short_name,  list(final_data[2].values()))
            axs[0,2].set_title(paper_names[2])
            axs[0,2].set_ylabel('Count')

            axs[1, 0].bar(categories_short_name, list(final_data[3].values()))
            axs[1, 0].set_title(paper_names[3])
            axs[1, 0].set_ylabel('Count')

            axs[1, 1].bar(categories_short_name, list(final_data[4].values()))
            axs[1, 1].set_title(paper_names[4])
            axs[1, 1].set_ylabel('Count')

            axs[1, 2].bar(categories_short_name, list(final_data[5].values()))
            axs[1, 2].set_title(paper_names[5])
            axs[1, 2].set_ylabel('Count')

            axs[2, 0].bar(categories_short_name, list(final_data[6].values()))
            axs[2, 0].set_title(paper_names[6])
            axs[2, 0].set_ylabel('Count')

            axs[2, 1].bar(categories_short_name, list(final_data[7].values()))
            axs[2, 1].set_title(paper_names[7])
            axs[2, 1].set_ylabel('Count')

            axs[2, 2].bar(categories_short_name, list(final_data[8].values()))
            axs[2, 2].set_title(paper_names[8])
            axs[2, 2].set_ylabel('Count')

            axs[3, 0].bar(categories_short_name, list(final_data[9].values()))
            axs[3, 0].set_title(paper_names[9])
            axs[3, 0].set_ylabel('Count')

            axs[3, 1].bar(categories_short_name, list(final_data[10].values()))
            axs[3, 1].set_title(paper_names[10])
            axs[3, 1].set_ylabel('Count')

            axs[3, 2].bar(categories_short_name, list(final_data[11].values()))
            axs[3, 2].set_title(paper_names[11])
            axs[3, 2].set_ylabel('Count')

            plt.show()

def additional_graphs(LLM_data, human_data):
    Menu = 10

    while Menu != 0:
        Menu = int(input(
            "1 for difference total references, 2 for comparison per paper , 3 for comparison per category, 0 to quit:      "))
        if Menu == 1:
            # Graph of the reference counts vs the name of the paper
            totals_human_data = human_data[-1][0::2]
            #This is for example [69, 69, 47, 58, 57, 74, 65, 53, 66, 65, 74, 69] because it is the metadata list
            totals_LLM_data = LLM_data[-1][0::2]
            difference_count = [totals_human_data[count]-totals_LLM_data[count] for count in range(len(totals_human_data))]
            print(difference_count, "difference count")
            # x_labels = human_data[-1][1::2] These are the names of the text files
            x_labels = ['AP1','AP2','INC1','INC2','LOAN1','LOAN2','O2C1','O2C2','P2P1','P2P2','EXP1','EXP2']

            # [1] takes the list with the neccesary data
            # [1::2] takes the first element and does skips of two (aka miss one out)

            plt.figure(figsize=(10, 5))
            plt.bar(x_labels, difference_count)

            plt.xlabel('File')
            plt.ylabel('$\Delta$ evaluated references')
            # plt.title('Difference in evaluated reference count (human total minus LLM total)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        if Menu == 2:
            # Based off Menu ==2 from the human evaluation

            fig, axs = plt.subplots(4, 3, figsize=(24, 21), squeeze=False, constrained_layout=True)

            categories_short_name = ['CP', 'SP', 'N-AR', 'N-AA', 'SMPR', 'None', 'Law']

            combined_y_human = [list(human_data[count].values()) for count in range((len(human_data) - 1))]
            y_label_human_cp = [list[0] for list in combined_y_human]
            y_label_human_sp = [list[1] for list in combined_y_human]
            y_label_human_nonar = [list[2] for list in combined_y_human]
            y_label_human_nonaa = [list[3] for list in combined_y_human]
            y_label_human_sm = [list[4] for list in combined_y_human]
            y_label_human_none = [list[5] for list in combined_y_human]
            y_label_human_law = [list[6] for list in combined_y_human]

            combined_y_LLM = [list(LLM_data[count].values()) for count in range((len(LLM_data) - 1))]
            y_label_LLM_cp = [list[0] for list in combined_y_LLM]
            y_label_LLM_sp = [list[1] for list in combined_y_LLM]
            y_label_LLM_nonar = [list[2] for list in combined_y_LLM]
            y_label_LLM_nonaa = [list[3] for list in combined_y_LLM]
            y_label_LLM_sm = [list[4] for list in combined_y_LLM]
            y_label_LLM_none = [list[5] for list in combined_y_LLM]
            y_label_LLM_law = [list[6] for list in combined_y_LLM]

            x = np.arange(7)
            # file_names = human_data[-1][1::2]
            file_names = ['AP_1','AP_2','INC_1','INC_2','LOAN_1','LOAN_2','O2C_1','O2C_2','P2P_1','P2P_2','EXP_1','EXP_2']
            x_labels = list(human_data[0].keys())
            #Replacing the 1,2,3,4,5,6,7 with the shortened names of the files (which are the keys of the data)


            axs[0, 0].bar(x - 0.1, list(human_data[0].values()), width=0.2, color='cyan')
            axs[0, 0].bar(x + 0.1, list(LLM_data[0].values()), width=0.2, color='orange')
            axs[0, 0].set_title(file_names[0])
            axs[0, 0].set_ylabel('Count')
            axs[0, 0].legend(["human", "LLM"])
            axs[0, 0].set_xticks(x,x_labels, rotation=45)


            axs[0, 1].bar(x - 0.1, list(human_data[1].values()), width=0.2, color='cyan')
            axs[0, 1].bar(x + 0.1, list(LLM_data[1].values()), width=0.2, color='orange')
            axs[0, 1].set_title(file_names[1])
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].legend(["human", "LLM"])
            axs[0, 1].set_xticks(x,x_labels, rotation=45)

            axs[0, 2].bar(x - 0.1, list(human_data[2].values()), width=0.2, color='cyan')
            axs[0, 2].bar(x + 0.1, list(LLM_data[2].values()), width=0.2, color='orange')
            axs[0, 2].set_title(file_names[2])
            axs[0, 2].set_ylabel('Count')
            axs[0, 2].legend(["human", "LLM"])
            axs[0, 2].set_xticks(x,x_labels, rotation=45)

            axs[1, 0].bar(x - 0.1, list(human_data[3].values()), width=0.2, color='cyan')
            axs[1, 0].bar(x + 0.1, list(LLM_data[3].values()), width=0.2, color='orange')
            axs[1, 0].set_title(file_names[3])
            axs[1, 0].set_ylabel('Count')
            axs[1, 0].legend(["human", "LLM"])
            axs[1, 0].set_xticks(x,x_labels, rotation=45)

            axs[1, 1].bar(x - 0.1, list(human_data[4].values()), width=0.2, color='cyan')
            axs[1, 1].bar(x + 0.1, list(LLM_data[4].values()), width=0.2, color='orange')
            axs[1, 1].set_title(file_names[4])
            axs[1, 1].set_ylabel('Count')
            axs[1, 1].legend(["human", "LLM"])
            axs[1, 1].set_xticks(x,x_labels, rotation=45)

            axs[1, 2].bar(x - 0.1, list(human_data[5].values()), width=0.2, color='cyan')
            axs[1, 2].bar(x + 0.1, list(LLM_data[5].values()), width=0.2, color='orange')
            axs[1, 2].set_title(file_names[5])
            axs[1, 2].set_ylabel('Count')
            axs[1, 2].legend(["human", "LLM"])
            axs[1, 2].set_xticks(x,x_labels, rotation=45)

            axs[2, 0].bar(x - 0.1, list(human_data[6].values()), width=0.2, color='cyan')
            axs[2, 0].bar(x + 0.1, list(LLM_data[6].values()), width=0.2, color='orange')
            axs[2, 0].set_title(file_names[6])
            axs[2, 0].set_ylabel('Count')
            axs[2, 0].legend(["human", "LLM"])
            axs[2, 0].set_xticks(x,x_labels, rotation=45)

            axs[2, 1].bar(x - 0.1, list(human_data[7].values()), width=0.2, color='cyan')
            axs[2, 1].bar(x + 0.1, list(LLM_data[7].values()), width=0.2, color='orange')
            axs[2, 1].set_title(file_names[7])
            axs[2, 1].set_ylabel('Count')
            axs[2, 1].legend(["human", "LLM"])
            axs[2, 1].set_xticks(x,x_labels, rotation=45)

            axs[2, 2].bar(x - 0.1, list(human_data[8].values()), width=0.2, color='cyan')
            axs[2, 2].bar(x + 0.1, list(LLM_data[8].values()), width=0.2, color='orange')
            axs[2, 2].set_title(file_names[8])
            axs[2, 2].set_ylabel('Count')
            axs[2, 2].legend(["human", "LLM"])
            axs[2, 2].set_xticks(x,x_labels, rotation=45)

            axs[3, 0].bar(x - 0.1, list(human_data[9].values()), width=0.2, color='cyan')
            axs[3, 0].bar(x + 0.1, list(LLM_data[9].values()), width=0.2, color='orange')
            axs[3, 0].set_title(file_names[9])
            axs[3, 0].set_ylabel('Count')
            axs[3, 0].legend(["human", "LLM"])
            axs[3, 0].set_xticks(x,x_labels, rotation=45)

            axs[3, 1].bar(x - 0.1, list(human_data[10].values()), width=0.2, color='cyan')
            axs[3, 1].bar(x + 0.1, list(LLM_data[10].values()), width=0.2, color='orange')
            axs[3, 1].set_title(file_names[10])
            axs[3, 1].set_ylabel('Count')
            axs[3, 1].legend(["human", "LLM"])
            axs[3, 1].set_xticks(x,x_labels, rotation=45)

            axs[3, 2].bar(x - 0.1, list(human_data[11].values()), width=0.2, color='cyan')
            axs[3, 2].bar(x + 0.1, list(LLM_data[11].values()), width=0.2, color='orange')
            axs[3, 2].set_title(file_names[11])
            axs[3, 2].set_ylabel('Count')
            axs[3, 2].legend(["human", "LLM"])
            axs[3, 2].set_xticks(x,x_labels, rotation=45)

            plt.show()

        if Menu == 3:
            # Based off Menu ==2 from above

            fig, axs = plt.subplots(4, 2, figsize=(25, 15), squeeze=False, constrained_layout=True)
            # Constrained layout: https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
            # squeeze: coordintates of barcharts
            # print(human_data[-1][1::2]) filenames (the x axis lables)
            papers_short_name = ['AP_1', 'AP_2', 'IT_1', 'IT_2', 'LA_1', 'LA_2', 'O2C_1', 'O2C_2', 'P2P_1', 'P2P_2', 'TE_1', 'TE_2']

            combined_y_human = [list(human_data[count].values()) for count in range((len(human_data) - 1))]
            y_label_human_cp = [list[0] for list in combined_y_human]
            y_label_human_sp = [list[1] for list in combined_y_human]
            y_label_human_nonar = [list[2] for list in combined_y_human]
            y_label_human_nonaa = [list[3] for list in combined_y_human]
            y_label_human_sm = [list[4] for list in combined_y_human]
            y_label_human_none = [list[5] for list in combined_y_human]
            y_label_human_law = [list[6] for list in combined_y_human]

            combined_y_LLM = [list(LLM_data[count].values()) for count in range((len(LLM_data) - 1))]
            y_label_LLM_cp = [list[0] for list in combined_y_LLM]
            y_label_LLM_sp = [list[1] for list in combined_y_LLM]
            y_label_LLM_nonar = [list[2] for list in combined_y_LLM]
            y_label_LLM_nonaa = [list[3] for list in combined_y_LLM]
            y_label_LLM_sm = [list[4] for list in combined_y_LLM]
            y_label_LLM_none = [list[5] for list in combined_y_LLM]
            y_label_LLM_law = [list[6] for list in combined_y_LLM]

            x = np.arange(12)
            # file_names = human_data[-1][1::2]
            x_labels = list(human_data[0].keys())
            #Replacing the 1,2,3,4,5,6,7 with the shortened names of the files (which are the keys of the data)

            axs[0, 0].bar(x - 0.1, y_label_human_cp, width=0.2, color='cyan')
            axs[0, 0].bar(x + 0.1, y_label_LLM_cp, width=0.2, color='orange')
            axs[0, 0].set_title('Company papers')
            axs[0, 0].set_ylabel('Count')
            axs[0, 0].legend(["human", "LLM"])
            axs[0, 0].set_xticks(x,papers_short_name, rotation=45)


            axs[0, 1].bar(x - 0.1, y_label_human_sp, width=0.2, color='cyan')
            axs[0, 1].bar(x + 0.1, y_label_LLM_sp, width=0.2, color='orange')
            axs[0, 1].set_title('Scientific papers')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].set_xticks(x,papers_short_name, rotation=45)

            axs[1, 0].bar(x - 0.1, y_label_human_nonar, width=0.2, color='cyan')
            axs[1, 0].bar(x + 0.1, y_label_LLM_nonar, width=0.2, color='orange')
            axs[1, 0].set_title('Non-acad. research')
            axs[1, 0].set_ylabel('Count')
            axs[1, 0].set_xticks(x, papers_short_name, rotation=45)

            axs[1, 1].bar(x - 0.1, y_label_human_nonaa, width=0.2, color='cyan')
            axs[1, 1].bar(x + 0.1, y_label_LLM_nonaa, width=0.2, color='orange')
            axs[1, 1].set_title('Non-acad. articles')
            axs[1, 1].set_ylabel('Count')
            axs[1, 1].set_xticks(x,papers_short_name, rotation=45)

            axs[2, 0].bar(x - 0.1, y_label_human_sm, width=0.2, color='cyan')
            axs[2, 0].bar(x + 0.1, y_label_LLM_sm, width=0.2, color='orange')
            axs[2, 0].set_title('Press/Social media')
            axs[2, 0].set_ylabel('Count')
            axs[2, 0].set_xticks(x,papers_short_name, rotation=45)

            axs[2, 1].bar(x - 0.1, y_label_human_none, width=0.2, color='cyan')
            axs[2, 1].bar(x + 0.1, y_label_LLM_none, width=0.2, color='orange')
            axs[2, 1].set_title('Not working link')
            axs[2, 1].set_ylabel('Count')
            axs[2, 1].set_xticks(x,papers_short_name, rotation=45)

            axs[3, 0].bar(x - 0.1, y_label_human_law, width=0.2, color='cyan')
            axs[3, 0].bar(x + 0.1, y_label_LLM_law, width=0.2, color='orange')
            axs[3, 0].set_title('Law documents')
            axs[3, 0].set_ylabel('Count')
            axs[3, 0].set_xticks(x,papers_short_name, rotation=45)

            axs[3, 1].bar(x - 0.1, human_data[-1][0::2], width=0.2, color='cyan')
            axs[3, 1].bar(x + 0.1, LLM_data[-1][0::2], width=0.2, color='orange')
            axs[3, 1].set_title('Total references')
            axs[3, 1].set_ylabel('Count')
            axs[3, 1].set_xticks(x,papers_short_name, rotation=45)



            plt.show()


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    Menu = 10

    while Menu != 0:
        Menu = int(input("1 for LLM domain knowledge analysis, 2 for LLM reference analysis, 0 to quit:      "))
        if Menu == 1:
            perform_DK_LLM_evaluation()

        if Menu ==2:

            while Menu != 0:
                Menu = int(input(
                    "1 for running LLM again, 2 for LLM vs human analysis graphs, 0 to quit:      "))
                if Menu == 1:
                    preprocessed_data = preprocess_references("References_LLM.txt")
                    perform_references_LLM_evaluation(preprocessed_data)
                if Menu == 2:
                    preprocessed_data = preprocess_references("All_references_classified.txt")
                    manually_processed_data = retreive_LLM_results()

                    additional_graphs(manually_processed_data, preprocessed_data)

def plotting_box_plot_results(files, graders, results):
    """
        This function uses the outputs from the main.py file to plot results.
         The (now redundant) function has been moved to this file to keep the main.py file clean.
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
