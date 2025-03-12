import os
import subprocess
import sys
import argparse
import glob
import re


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--convert_logs_dir", type=str, required=True)
    parser.add_argument("--container_image", type=str, required=True)
    parser.add_argument("--opensci_megatron_path", type=str, required=True, help="Path to Megatron-LM-Open-Sci")
    parser.add_argument("--megatron_path", type=str, required=True, help="Path to Megatron-LM")
    parser.add_argument("--open_sci_hf_path", type=str, required=True, help="Path to open-sci-hf")
    parser.add_argument("--train_logs_dir_or_path", type=str, required=True, help="Path to train logs directory or file")
    parser.add_argument("--save_checkpoints_dir", type=str, required=True, help="Path to save checkpoints directory")

    args = parser.parse_args()
    account = args.account
    partition = args.partition
    convert_logs_dir = args.convert_logs_dir
    container_image = args.container_image
    opensci_megatron_path = args.opensci_megatron_path
    megatron_path = args.megatron_path
    open_sci_hf_path = args.open_sci_hf_path
    train_logs_dir_or_path = args.train_logs_dir_or_path
    save_checkpoints_dir = args.save_checkpoints_dir

    os.makedirs(convert_logs_dir, exist_ok=True)
    os.makedirs(save_checkpoints_dir, exist_ok=True)

    sbatch_template_path = os.path.join(opensci_megatron_path, "scripts/ckpt/convert_full/template.sbatch")
    with open(sbatch_template_path, "r") as f:
        sbatch_template = f.read()
        # escape ${} in f-strings with double curly braces
        
        # escape cat <<EOF > ../config.json\n{...}\nEOF
        cat_eof_data = re.search(r"cat <<EOF.*?EOF", sbatch_template, re.DOTALL).group()
        sbatch_template = sbatch_template.replace(cat_eof_data, "<cat_eof_data>")
        sbatch_template = re.sub(r"\$\{(.+?)\}", r"\${{\1}}", sbatch_template).replace("\$", "$")

    if "::" in train_logs_dir_or_path:
        train_logs = train_logs_dir_or_path.split("::")
    elif os.path.isdir(train_logs_dir_or_path):
        train_logs = glob.glob(os.path.join(train_logs_dir_or_path, "*.out"))
    else:
        train_logs = [train_logs_dir_or_path]

    if len(train_logs) == 0:
        print(f"No train logs found in {train_logs_dir_or_path}")
        sys.exit(1)

    for train_log in train_logs:
        print(f"Converting {os.path.basename(train_log)}")

        sbatch_script = sbatch_template.format(
            account=account,
            partition=partition,
            container_image=container_image,
            opensci_megatron_path=opensci_megatron_path,
            megatron_path=megatron_path,
            open_sci_hf_path=open_sci_hf_path,
            train_logs_path=train_log,
            save_checkpoints_dir=save_checkpoints_dir,
            convert_logs_dir=convert_logs_dir
        )

        sbatch_script = sbatch_script.replace("<cat_eof_data>", cat_eof_data)

        sbatch_script_path = os.path.join(convert_logs_dir, f"convert_{os.path.basename(train_log)}.sbatch")
        with open(sbatch_script_path, "w") as f:
            f.write(sbatch_script)

        subprocess.run(["sbatch", sbatch_script_path])
        print(f"Submitted {sbatch_script_path}")

if __name__ == "__main__":
    main()