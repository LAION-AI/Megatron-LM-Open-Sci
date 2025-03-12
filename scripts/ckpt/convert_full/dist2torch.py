import sys
import os
import argparse
import re
import json



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-path", type=str, default="logs.txt")
    parser.add_argument("--torch-chpt-path", type=str, default="torch_chpt")
    parser.add_argument("--dist-chpt-path", type=str, default="dist_chpt")
    parser.add_argument("--tensorboard-dir", type=str, default="~/tensorboard")
    parser.add_argument("--data-cache-path", type=str, default="~/data_cache")

    args = parser.parse_args()
    logs_path = args.logs_path
    torch_chpt_path = args.torch_chpt_path
    dist_chpt_path =args.dist_chpt_path
    tensorboard_dir = args.tensorboard_dir
    data_cache_dir = args.data_cache_path

    os.makedirs(torch_chpt_path, exist_ok=True)
    os.makedirs(dist_chpt_path, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(data_cache_dir, exist_ok=True)
    with open(logs_path, "r") as f:
        logs = f.read()
        pretrain_cmd = re.search(r"pretrain_gpt.py.*", logs).group()
    
    pretrain_cmd += f"--ckpt-convert-save {torch_chpt_path}"
    pretrain_cmd_dict = {}
    pretrain_cmd = pretrain_cmd.replace("pretrain_gpt.py", "").split(" ")
    for i in range(len(pretrain_cmd)):
        if pretrain_cmd[i].startswith("--"):
            pretrain_cmd_dict[pretrain_cmd[i].strip()] = pretrain_cmd[i+1] if not pretrain_cmd[i+1].startswith("--") else ""

    pretrain_cmd_dict.pop("--tensorboard-dir")
    pretrain_cmd_dict["--tensorboard-dir"] = tensorboard_dir
    pretrain_cmd_dict["--data-cache-path"] = data_cache_dir
    pretrain_cmd_dict["--save"] = dist_chpt_path
    pretrain_cmd_dict["--ckpt-convert-format"] = "torch"
    pretrain_cmd_dict["--ckpt-convert-save"] = torch_chpt_path
    pretrain_cmd = " ".join([f"{k} {v}" for k, v in pretrain_cmd_dict.items()])
    pretrain_cmd = "pretrain_gpt.py " + pretrain_cmd

    print(pretrain_cmd)

if __name__ == "__main__":
    main()