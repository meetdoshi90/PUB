import os
import json

output_file = "output.csv"
# new_tasks = [1,2,9,10,13]
# old_tasks = [0,3,4,5,6,7,8,11,14]
# target_dirs = {"llama-2-7b", "llama-2-7b-chat", "llama-2-13b","llama-2-13b-chat","llama-2-70b","llama-2-70b-chat","t5-11b"}
# with open(output_file, "w") as output:
#     for root, dirs, files in os.walk("/raid/nlp/pranavg/iclr/Results/metrics/task_3/"):
#         if os.path.basename(root) in target_dirs:
#             for file in files:
#                 # print(file)
#                 if file.endswith(".csv") and  ("mcqa" in file and "few_shot" in file):
#                     file_path = os.path.join(root, file)
#                     with open(file_path, "r") as csvfile:
#                         content = csvfile.read()
#                         data = json.loads(content)

#                         if "f1_macro" in data:
#                             accuracy_value = data["accuracy"]
#                             subfolder_name = os.path.basename(root)
#                             output.write(f"{subfolder_name}-{file},{accuracy_value}\n")
import os
import json

# Define the output file path
output_file = "output_file.csv"

# Create or open the output file
with open(output_file, "w") as output:
    # Iterate through the task subfolders
    for task_folder in range(0, 15):  # Assuming task folders are named task1, task2, ..., task14
        task_folder_name = f"task_{task_folder}"
        task_path = f"/raid/nlp/pranavg/iclr/Results_backup/metrics/{task_folder_name}/"

        target_dirs = {
            "llama-2-7b", "llama-2-7b-chat", "llama-2-13b", 
            "llama-2-13b-chat", "llama-2-70b", "llama-2-70b-chat", "t5-11b", "flan-t5-xxl"
        }

        # Iterate through each subfolder for the current task
        for root, dirs, files in os.walk(task_path):
            if os.path.basename(root) in target_dirs:
                for file in files:
                    if file.endswith(".csv") and ("mcqa" in file and "few_shot_k3_run0" in file):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as csvfile:
                            content = csvfile.read()
                            data = json.loads(content)

                            if "f1_macro" in data:
                                accuracy_value = data["f1_macro"]
                                subfolder_name = os.path.basename(root)
                                output.write(f"{task_folder_name}-{subfolder_name}-{file},{accuracy_value}\n")
