#!/usr/bin/env python
"""
Backport pull requests to a particular branch.

Usage: backport_pr.py branch [PR]

e.g.:

    python tools/backport_pr.py 0.13.1 123

to backport PR #123 onto branch 0.13.1

or

    python tools/backport_pr.py 1.x

to see what PRs are marked for backport that have yet to be applied.

Copied from IPython 9e82bc5
https://github.com/ipython/ipython/blob/master/tools/backport_pr.py
"""

import os
import re
import sys

from subprocess import Popen, PIPE, check_call, check_output
from urllib import urlopen

from gh_api import (
    get_issues_list,
    get_pull_request,
    get_pull_request_files,
    is_pull_request,
    get_milestone_id,
)

from pandas import Series


def find_rejects(root='.'):
    for dirname, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith('.rej'):
                yield os.path.join(dirname, fname)


def get_current_branch():
    branches = check_output(['git', 'branch'])
    for branch in branches.splitlines():
        if branch.startswith('*'):
            return branch[1:].strip()


def backport_pr(branch, num, project='statsmodels/statsmodels'):
    current_branch = get_current_branch()
    if branch != current_branch:
        check_call(['git', 'checkout', branch])
    check_call(['git', 'pull'])
    pr = get_pull_request(project, num, auth=True)
    files = get_pull_request_files(project, num, auth=True)
    patch_url = pr['patch_url']
    title = pr['title']
    description = pr['body']
    fname = "PR%i.patch" % num
    if os.path.exists(fname):
        print("using patch from {fname}".format(**locals()))
        with open(fname) as fd:
            patch = fd.read()
    else:
        req = urlopen(patch_url)
        patch = req.read()

    msg = "Backport PR #%i: %s" % (num, title) + '\n\n' + description
    check = Popen(['git', 'apply', '--check', '--verbose'], stdin=PIPE)
    a, b = check.communicate(patch)

    if check.returncode:
        print("patch did not apply, saving to {fname}".format(**locals()))
        print("edit {fname} until `cat {fname} | git apply --check` succeeds"
              .format(**locals()))
        print("then run tools/backport_pr.py {num} again".format(**locals()))
        if not os.path.exists(fname):
            with open(fname, 'wb') as fd:
                fd.write(patch)
        return 1

    p = Popen(['git', 'apply'], stdin=PIPE)
    a, b = p.communicate(patch)

    filenames = [f['filename'] for f in files]

    check_call(['git', 'add'] + filenames)

    check_call(['git', 'commit', '-m', msg])

    print("PR #%i applied, with msg:" % num)
    print()
    print(msg)
    print()

    if branch != current_branch:
        check_call(['git', 'checkout', current_branch])

    return 0


backport_re = re.compile(r"[Bb]ackport.*?(\d+)")


def already_backported(branch, since_tag=None):
    """return set of PRs that have been backported already"""
    if since_tag is None:
        btag = check_output(['git', 'describe', branch, '--abbrev=0'])
        since_tag = btag.decode('utf8').strip()
    cmd = ['git', 'log', '%s..%s' % (since_tag, branch), '--oneline']
    lines = check_output(cmd).decode('utf8')
    return set(int(num) for num in backport_re.findall(lines))


def should_backport(labels=None, milestone=None):
    """return set of PRs marked for backport"""
    if labels is None and milestone is None:
        raise ValueError("Specify one of labels or milestone.")
    elif labels is not None and milestone is not None:
        raise ValueError("Specify only one of labels or milestone.")
    if labels is not None:
        issues = get_issues_list("statsmodels/statsmodels",
                                 labels=labels,
                                 state='closed',
                                 auth=True)
    else:
        milestone_id = get_milestone_id("statsmodels/statsmodels",
                                        milestone,
                                        auth=True)
        issues = get_issues_list("statsmodels/statsmodels",
                                 milestone=milestone_id,
                                 state='closed',
                                 auth=True)

    should_backport = []
    merged_dates = []
    for issue in issues:
        if not is_pull_request(issue):
            continue
        pr = get_pull_request("statsmodels/statsmodels", issue['number'],
                              auth=True)
        if not pr['merged']:
            print("Marked PR closed without merge: %i" % pr['number'])
            continue
        if pr['number'] not in should_backport:
            merged_dates.append(pr['merged_at'])
            should_backport.append(pr['number'])
    return Series(merged_dates, index=should_backport)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) < 3:
        branch = sys.argv[1]
        already = already_backported(branch)
        # NOTE: change this to the label you have used for marking a backport
        should = should_backport(milestone="0.5.1")
        print("The following PRs should be backported:")
        to_backport = []
        if already:
            should = should.loc[set(should.index).difference(already)]
        should.sort()
        for pr, date in should.iteritems():
            print(pr)
        sys.exit(0)

    sys.exit(backport_pr(sys.argv[1], int(sys.argv[2])))
