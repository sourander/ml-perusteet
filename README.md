# Skeleton for Documentation Project

## Dependencies
* Python >=3.10
* Python Poetry

## How to use

To create a new documentation project using this skeleton, do the following:

### 1. Instance template repository

Create a repository from this template. This is guided in the [GitHub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).


### 2. Clone locally and set up Python Env

You need to have Python (`3.10` or newer) installed. You also need to have [Python Poetry installed](https://python-poetry.org/docs/#installation).

```bash
# Clone
git clone 'new-repo-url'

# Update your Python Poetry and install dependencies
poetry self update
poetry install
```

### 3. Define the LICENSE

Use the formula in the `LICENSE.template` file. Fill in the Jinja-like slots (e.g. `{{ project }} => Project's Name`) and rename the file to `LICENSE`.

### 4. Define the copyright info also into site

Edit the following block to contain the information you need:

```yaml
# TODO: Replace the copyright with correct. Also add LICENCE file. Template included.
copyright: |
  Copyright &copy; 2023 <a href="https://www.example.com">Example Organization</a>. 
  Licenced under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">XX-XX-XX 4.0</a>
```

### 5. Modify Docs

Modify the contents of `docs/`. You `.pages` files if you need other than `[a-z]` ordering for your MarkDown files or, say, a navigation item that contains special characters (e.g. `TCP/IP`) or custom capitalization (e.g. `LaTeX `).

File naming: Use only ASCII characters and underscores in file names.

## 6. Run local build

To build the project locally to `site/` and to run a web server, you can simply run `poetry run mkdocs serve`. Visit the site at `localhost:8080`.

### 7. Build in GitHub Pages

Merge to `master` branch in any Repository. The scripts at `.github/workflows/` will be executed by Github Actions.

Sadly, GitHub will not automatically publish your Pages as could be expected. You need to visit the **Settings | 
Pages** (at `https://github.com/<username>/<reponame>/settings/pages`). There, under heading **Build and 
deployment**, choose Branch as `gh-pages` and path as `/ (root)`. Click Save. From now on, your Pages should be 
updated whenever you push to master. You should see a workflow with a title *pages build and deployment* in your 
Actions after each push.

## Batteries Included?

This template comes with two plugins: 
* [MkDocs static i18n](https://github.com/ultrabug/mkdocs-static-i18n) for multilanguage
* [MkDocs Awesome Pages](https://github.com/lukasgeiter/mkdocs-awesome-pages-plugin) for page ordering


## Need support for multilanguage?

Add the following to the `mkdocs.yml` file:

```yaml
plugins:
  - i18n:
      docs_structure: suffix
      default_language: en
      languages:
        en:
          name: English
          build: true
        fi:
          name: Finnish
          build: true
      # You can translate navigation items here if need be
      nav_translations:
        fi:
          Some Headline: Jokin otsikko
```

Add and install dependency: `poetry add mkdocs-static-i18n`. After this, you are ready to add alternative language files. Check [i18n Docs](https://github.com/ultrabug/mkdocs-static-i18n) for further information.

**NOTE:** You will need to modify the `.pages` files if you are using both Awesome Pages and i18n together. Example below.

```yaml
nav:
    - index.md         # Without i18n
    - ... | index*.md  # With i18n
```



## How to access

The URI for the GitHub Pages is `https://<username>.github.io/<repo_name>/`. The *pages build and deployment* 
workflow will output a link to this page.

## How to squash history

Let's say you have started working on a repository created using this template repo. In the beginning, your repo has been Private, and now you want to make it Public. You want to make sure there is nothing in the history that would, say, violate copyrights or include passphrases. To do this, you could squash all the commits into one. First, make sure you have committed your changes to origin **and have backed up your project**. You may lose your work if you screw up. You have been warned. To do this, do:

```sh
# Create a new orphan branch named 'new-master' based on the 'master' branch.
# An orphan branch is a new branch that has no commit history from the source branch.
git checkout --orphan new-master master

# Create a new initial commit for the orphan branch, providing a commit message.
# This commit serves as the starting point for your squashed commit history.
git commit -m "Squashed all commits"

# Overwrite the reference of the 'master' branch with the new orphan branch,
# effectively replacing the old history with the new commit you just created.
git branch -M new-master master

# Forcefully push the changes to the remote repository.
# This overwrites the remote 'master' branch with your new history.
git push --force
```

If you have the project cloned on other machines, they will need to either remove the directory and clone the project again, OR, run the following commands:

```sh
git fetch origin
git reset --hard origin/master
```