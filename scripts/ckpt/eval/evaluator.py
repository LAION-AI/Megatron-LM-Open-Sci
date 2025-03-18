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
    parser.add_argument("--eval_logs_dir", type=str, required=True)
    parser.add_argument("--eval_save_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--checkpoints_dir_or_path", type=str, required=True)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--opensci_megatron_path", type=str, required=True, help="Path to Megatron-LM-Open-Sci")
    parser.add_argument("--eval_cache_dir", type=str, default=None)

    args = parser.parse_args()
    account = args.account
    partition = args.partition
    eval_logs_dir = args.eval_logs_dir
    eval_save_dir = args.eval_save_dir
    tasks = args.tasks
    batch_size = args.batch_size
    checkpoints_dir_or_path = args.checkpoints_dir_or_path
    hf_cache_dir = args.hf_cache_dir or os.environ.get("HF_HOME")
    opensci_megatron_path = args.opensci_megatron_path

    os.makedirs(eval_logs_dir, exist_ok=True)
    os.makedirs(eval_save_dir, exist_ok=True)

    sbatch_template_path = os.path.join(opensci_megatron_path, "scripts/ckpt/eval/template.sbatch")
    with open(sbatch_template_path, "r") as f:
        sbatch_template = f.read()
        # escape ${} in f-strings with double curly braces
        sbatch_template = re.sub(r"\$\{(.+?)\}", r"\${{\1}}", sbatch_template).replace("\$", "$")
    if "::" in checkpoints_dir_or_path:
        checkpoints = checkpoints_dir_or_path.split("::")
    elif len(glob.glob(os.path.join(checkpoints_dir_or_path,"*.safetensors"))) > 0:
        checkpoints = [checkpoints_dir_or_path]
    else:
        checkpoints = glob.glob(os.path.join(checkpoints_dir_or_path, "*"))

    if len(checkpoints) == 0:
        print(f"No checkpoints found in {checkpoints_dir_or_path}")
        sys.exit(1)

    for checkpoint in checkpoints:
        print(f"Evaluating {os.path.basename(checkpoint)}")

        sbatch_script = sbatch_template.format(
            account=account,
            partition=partition,
            eval_logs_dir=eval_logs_dir,
            eval_save_dir=eval_save_dir,
            tasks=tasks,
            batch_size=batch_size,
            model_path=checkpoint,
            hf_cache_dir=hf_cache_dir,
            cache_dir=args.eval_cache_dir
        )

        sbatch_script_path = os.path.join(eval_logs_dir, f"{os.path.basename(checkpoint)}.sbatch")
        with open(sbatch_script_path, "w") as f:
            f.write(sbatch_script)

        subprocess.run(["sbatch", sbatch_script_path])
        break

if __name__ == "__main__":
    main()