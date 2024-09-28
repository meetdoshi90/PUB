# PUB: A Pragmatics Understanding Benchmark for Assessing LLMs’ Pragmatics Capabilities

This repository contains the code for the paper published at ACL 2024

The dataset can be found at [cfilt/PUB](https://huggingface.co/datasets/cfilt/PUB)

## Getting started

> Main.py is the main file to evaluate models. You can also add models in the same file for inference.

> Pragmatics/ folder consists of prompt selection and close/mcqa prompt evaluation code.

> Human_eval.ipynb and Human_results.ipynb are the files that were used to calculate performance of Humans on the PUB dataset.

> Analysis/ folder consists of error analysis done in the paper.

> Gpt3_results_*/ folder consists of gpt3 evaluation code.

## Citation

```
@inproceedings{sravanthi-etal-2024-pub,
    title = "{PUB}: A Pragmatics Understanding Benchmark for Assessing {LLM}s{'} Pragmatics Capabilities",
    author = "Sravanthi, Settaluri  and
      Doshi, Meet  and
      Tankala, Pavan  and
      Murthy, Rudra  and
      Dabre, Raj  and
      Bhattacharyya, Pushpak",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.719",
    doi = "10.18653/v1/2024.findings-acl.719",
    pages = "12075--12097",
}
```