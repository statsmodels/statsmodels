"""
Generate a release note template.  The key parameters are
RELEASE, VERSION, MILESTONE BRANCH, and LAST_COMMIT_SHA.

LAST_COMMIT_SHA is the sha of the commit used to produce the previous
version. This is used to determine the time stamp where commits in the
current release begin.

Requires PyGitHub, dateparser and jinja2
"""

from collections import defaultdict
import datetime as dt
import os

import dateparser
from github import Github
from jinja2 import Template

# Full release version
RELEASE = "0.12.0"
# The current milestone and short version
VERSION = MILESTONE = "0.12"
# This is the final commit from the previous release
LAST_COMMIT_SHA = "e11c4e45037cfe931f8d3ca4df06e2ec818175b1"
# Branch, usually master but can be a maintenance branch as well
BRANCH = "master"
# Provide access token using command line to keep out of repo
ACCESS_TOKEN = os.environ.get("GITHUB_ACCESS_TOKEN", None)
if not ACCESS_TOKEN:
    raise RuntimeError("Must set environment variable GITHUB_ACCESS_TOKEN "
                       "containing a valid GitHub access token before running"
                       "this program.")

# Using an access token
g = Github(ACCESS_TOKEN)
# Get the repo
statsmodels = g.get_user("statsmodels").get_repo("statsmodels")
# Look up the modification time of the commit used to tag the previous release
last_modified = statsmodels.get_commit(LAST_COMMIT_SHA).commit.last_modified
last_modified = dateparser.parse(last_modified)
# Look for times creater than this time plus 1 second
first_commit_time = last_modified + dt.timedelta(seconds=1)
first_commit_time_iso = first_commit_time.isoformat()

# General search for sm/sm, PR, merged, merged> first commit time and branch
query_parts = ("repo:statsmodels/statsmodels",
               "is:pr",
               "is:merged",
               "merged:>{}".format(first_commit_time_iso),
               "base:{}".format(BRANCH))
query = " ".join(query_parts)
merged_pull_data = []
merged_pulls = g.search_issues(query)

# Get the milestone for the current release or create if it does not exist
milestone = None
for ms in statsmodels.get_milestones():
    if ms.title == MILESTONE:
        milestone = ms
if milestone is None:
    description = "Release {} issues and pull requests".format(MILESTONE)
    milestone = statsmodels.create_milestone(MILESTONE, state="open",
                                             description=description)

# Get PR data and set the milestone if needed
for pull in merged_pulls:
    merged_pull_data.append({"number": pull.number,
                             "title": pull.title,
                             "login": pull.user.login,
                             "labels": pull.labels,
                             "milestone": pull.milestone}
                            )
    if pull.milestone is None or pull.milestone != milestone:
        pull.edit(milestone=milestone)

merged_pull_data = sorted(merged_pull_data, key=lambda x: x["number"])

# Robust name resolutions using commits and GitHub lookup
names = defaultdict(set)
extra_names = set()
for pull in merged_pull_data:
    print("Reading commit data for PR#{}".format(pull["number"]))
    pr = statsmodels.get_pull(pull["number"])
    for commit in pr.get_commits():
        name = commit.commit.author.name
        if name and commit.author:
            names[commit.author.login].update([name])
        elif name:
            extra_names.update([name])

for login in names:
    user = g.get_user(login)
    if user.name:
        names[login].update([user.name])

# Continue trying to resolve to human names
contributors = []
for login in names:
    print("Reading user data for {}".format(login))
    user_names = list(names[login])
    if len(user_names) == 1:
        name = user_names[0]
        if " " in name:
            name = name.title()
        contributors.append(name)
    else:
        valid = [name for name in user_names if " " in name]
        if len(valid) == 0:
            contributors.append(login)
        else:
            contributors.append(valid[0].title())

contributors = sorted(set(contributors))

# Get all issues closed since first_commit_time_iso
query_parts = ("repo:statsmodels/statsmodels",
               "is:issue",
               "is:closed",
               "closed:>{}".format(first_commit_time_iso))
query = " ".join(query_parts)
closed_issues = g.search_issues(query)
# Set the milestone for these issues if needed
for issue in closed_issues:
    if issue.milestone is None or issue.milestone != milestone:
        issue.edit(milestone=milestone)

issues_closed = closed_issues.totalCount

# Create a What's New Dictionary to automatically populate the template
# Structure is dict[module, dict[pr number, sanitized title]]
whats_new = defaultdict(dict)
for pull in merged_pull_data:
    if pull["labels"]:
        labels = [lab.name for lab in pull["labels"] if
                  not lab.name.startswith("type")]
        if "maintenance" in labels and len(labels) > 1:
            labels.remove("maintenance")
        elif "comp-docs" in labels and len(labels) > 1:
            labels.remove("comp-docs")
        for label in labels:
            label = label.split("comp-")[-1].replace("-", ".")
            number = pull["number"]
            title = pull["title"]
            if ": " in title:
                title = ": ".join(title.split(": ")[1:])
            title = title[:1].upper() + title[1:]
            whats_new[label][number] = title

whats_new = {key: whats_new[key] for key in sorted(whats_new)}

# Variables for the template
variables = {"milestone": MILESTONE,
             "release": RELEASE,
             "version": VERSION,
             "issues_closed": issues_closed,
             "pulls_merged": len(merged_pull_data),
             "contributors": contributors,
             "pulls": merged_pull_data,
             "whats_new": whats_new,
             }

# Read the template and generate the output
with open("release_note.tmpl", encoding="utf-8") as tmpl:
    tmpl_data = tmpl.read()
    t = Template(tmpl_data)
    rendered = t.render(**variables)
    file_name = "version{}.rst".format(VERSION)
    with open(file_name, encoding="utf-8", mode="w") as out:
        out.write(rendered)
