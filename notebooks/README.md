# Notebooks for Course

## Naming Convention

This directory contains Marimo Notebooks used in the course. Unlike the theory site, these Notebooks are written in English, not in Finnish. Each notebook corresponds to a specific topic or module covered in the curriculum. If you check the `docs/.nav.yml`, you will notice that each course category/week has a hundred-starting identifier:

```yaml
nav:
    - index.md
    - neuroverkot    # 100
    - tensorit       # 200
    - ...
    - kieli          # 700
    - aikasarjat     # 800
```

Underneath each category, each entry will have their own ten-starting identifier in the header part of the Markdown file, like the first entry in `neuroverkot` category is 100, and the next one is 110, and so on. For example, the `docs/neuroverkot/neuroverkot_101.md` file will contain a header with a value `priority: 101`. Just go and check the files to see how it works, if you are curious.

The Notebooks will match to this numbering scheme. For example, if the lesson `docs/neuroverkot/syvaoppiminen_FC.md` has a priority of `110`, and the lesson would have three Notebooks, they would be named as follows:

```
notebooks/nb/100/110_lorem.py
notebooks/nb/100/111_ipsum.py
notebooks/nb/100/112_dolor.py
```

## Using Marimo

The `uv` project exists so that you can use Marimo either in Browser or using VS Code Extension for Marimo. This guide focuses on the browser usage, but there is a section about VS Code as well. Note that teacher will use it in the browser during the lessons. On this course, we are not using GPU (like in Syväoppiminen I course), so there is no mandatory requirements to containerize the notebook runtime. You are free to do so if you will, but running it locally is just fine:

1. Find a way to install `uv` on your system (guide: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/))
2. Reboot (or at least close all terminals and VS Code and open a new one)
3. Make a copy of this `notebook/` directory to wherever you want

Then, simply:

```bash
cd notebook
uv run marimo edit
```

The Marimo will open into your browser. Find the Notebook you need and have fun learning.


## Teacher 👨‍🏫: How to handle Solutions

Some of the Notebooks contain exercises. Example solutions are in a subdirectory `solutions`. Old hack was backing them up to OneDrive using a script. New solution is using `git-crypt` to encrypt the solutions directory, so that only teachers with the decryption key can access them.

Here is a guide for setup. I might move this guide to `How to Git` one day. For now, it is here.

### Prerequisites

The `git-crypt` key has been originally created only on one machine and then distributed to other machines utilizing a password manager.

Here we assume that: 

1. the `.gitignore` file contains a filter that we need. Check the file in `../.gitignore` for details.
2. the `gh-solutions.git-crypt.key` has been downloaded to `$HOME` directory.

The key file is copied into `.git/git-crypt/keys/default` after running the commands below. Run this only once per repository (or again if you need to clone the repository again).

```bash
# Navigate to whereever the repo root is
cd ~/Code/sourander/ml-perusteet

# Unlock
git-crypt unlock ~/gh-solutions.git-crypt.key

# Remove the key
rm ~/gh-solutions.git-crypt.key

# Check
git-crypt status -e
```

This should list all files in `notebooks/nb/solutions` as encrypted.
