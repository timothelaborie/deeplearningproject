# Generate a bash file for the hyper-parameter search
dataset = "cifar10"
seeds = list([42, 4711, 314159])

default_parameters = {"dataset": [dataset], "optim": ["sgd"], "epochs": [270], "batch_size": [32], "learning_rate": [0.1], "momentum": [0.9], "weight_decay": [1e-4], "gamma":[0.1], "augment": ["none", "cifar10"],  "random_seed": seeds}
standard_parameters = {"variant": ["standard"]}
mixup_parameters = {"variant": ["mixup"]}
manifold_mixup_parameters = {"variant": ["manifold_mixup"]}
gan_parameters = {"variant": ["mixup_gan"]}

for key in default_parameters.keys():
    standard_parameters[key] = default_parameters[key]
    mixup_parameters[key] = default_parameters[key]
    manifold_mixup_parameters[key] = default_parameters[key]
    gan_parameters[key] = default_parameters[key]


def dict_to_command(command_dict):
    commands = ["sbatch --time 5-0 --ntasks=4 --mem-per-cpu=8G --gpus=rtx_2080_ti:1  --wrap \"PYTHONPATH=/cluster/home/bgunders/deeplearningproject /cluster/scratch/bgunders/conda_envs/sg3exl/bin/python3.9 main.py"]
    for argument in command_dict:
        commands = ["{} --{} {}".format(old_command, argument, argument_instance) for old_command in commands for argument_instance in command_dict[argument]]
    new_commands = []
    for command in commands:
        new_commands.append(command + "\"")
    return new_commands

all_commands = dict_to_command(gan_parameters)
all_commands += dict_to_command(mixup_parameters)
all_commands += dict_to_command(manifold_mixup_parameters)
all_commands += dict_to_command(standard_parameters)

file_name = "start_search.sh"
f = open(file_name, "w")
f.write("#!/bin/bash\nmodule load gcc/8.2.0 python_gpu/3.9.9\n\nexport HTTP_PROXY=http://proxy.ethz.ch:3128\nexport HTTPs_PROXY=http://proxy.ethz.ch:3128\n\n")
for command in all_commands:
    f.write(command + "\n")
f.close()
print("The {} commands are stored in {}".format(len(all_commands), file_name))