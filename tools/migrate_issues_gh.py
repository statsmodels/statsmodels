#!/usr/bin/env python
"""Launchpad to github bug migration script.

There's a ton of code from Hydrazine copied here:
https://launchpad.net/hydrazine


Usage
-----

This code is meant to port a bug database for a project from Launchpad to
GitHub. It was used to port the IPython bug history.

The code is meant to be used interactively. I ran it multiple times in one long
IPython session, until the data structures I was getting from Launchpad looked
right. Then I turned off (see 'if 0' markers below) the Launchpad part, and ran
it again with the github part executing and using the 'bugs' variable from my
interactive namespace (via"%run -i" in IPython).

This code is NOT fire and forget, it's meant to be used with some intelligent
supervision at the wheel. Start by making a test repository (I made one called
ipython/BugsTest) and upload only a few issues into that. Once you are sure
that everything is OK, run it against your real repo with all your issues.

You should read all the code below and roughly understand what's going on
before using this. Since I didn't intend to use this more than once, it's not
particularly robust or documented. It got the job done and I've never used it
again.

Configuration
-------------

To pull things off LP, you need to log in first (see the Hydrazine docs). Your
Hydrazine credentials will be cached locally and this script can reuse them.

To push to GH, you need to set below the GH repository owner, API token and
repository name you wan to push issues into. See the GH section for the
necessary variables.
"""

import collections
import os.path
import subprocess
import sys
import time

from pprint import pformat

import launchpadlib
from launchpadlib.credentials import Credentials
from launchpadlib.launchpad import (
    Launchpad, STAGING_SERVICE_ROOT, EDGE_SERVICE_ROOT )

#-----------------------------------------------------------------------------
# Launchpad configuration
#-----------------------------------------------------------------------------
# The official LP project name
PROJECT_NAME = 'statsmodels'

# How LP marks your bugs, I don't know where this is stored, but they use it to
# generate bug descriptions and we need to split on this string to create
# shorter Github bug titles
PROJECT_ID = 'statsmodels'

# Default Launchpad server, see their docs for details
service_root = EDGE_SERVICE_ROOT

#-----------------------------------------------------------------------------
# Code copied/modified from Hydrazine (https://launchpad.net/hydrazine)
#-----------------------------------------------------------------------------

# Constants for the names in LP of certain
lp_importances = ['Critical', 'High', 'Medium', 'Low', 'Wishlist', 'Undecided']

lp_status = ['Confirmed', 'Triaged', 'Fix Committed', 'Fix Released',
             'In Progress',"Won't Fix", "Incomplete", "Invalid", "New"]

def squish(a):
    return a.lower().replace(' ', '_').replace("'",'')

lp_importances_c = set(map(squish, lp_importances))
lp_status_c = set(map(squish, lp_status))

def trace(s):
    sys.stderr.write(s + '\n')


def create_session():
    lplib_cachedir = os.path.expanduser("~/.cache/launchpadlib/")
    hydrazine_cachedir = os.path.expanduser("~/.cache/hydrazine/")
    rrd_dir = os.path.expanduser("~/.cache/hydrazine/rrd")
    for d in [lplib_cachedir, hydrazine_cachedir, rrd_dir]:
        if not os.path.isdir(d):
            os.makedirs(d, mode=0700)

    hydrazine_credentials_filename = os.path.join(hydrazine_cachedir,
        'credentials')
    if os.path.exists(hydrazine_credentials_filename):
        credentials = Credentials()
        credentials.load(file(
            os.path.expanduser("~/.cache/hydrazine/credentials"),
            "r"))
        trace('loaded existing credentials')
        return Launchpad(credentials, service_root,
            lplib_cachedir)
        # TODO: handle the case of having credentials that have expired etc
    else:
        launchpad = Launchpad.get_token_and_login(
            'Hydrazine',
            service_root,
            lplib_cachedir)
        trace('saving credentials...')
        launchpad.credentials.save(file(
            hydrazine_credentials_filename,
            "w"))
        return launchpad

def canonical_enum(entered, options):
    entered = squish(entered)
    return entered if entered in options else None

def canonical_importance(from_importance):
    return canonical_enum(from_importance, lp_importances_c)

def canonical_status(entered):
    return canonical_enum(entered, lp_status_c)

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

class Base(object):
    def __str__(self):
        a = dict([(k,v) for (k,v) in self.__dict__.iteritems()
                  if not k.startswith('_')])
        return pformat(a)

    __repr__ = __str__


class Message(Base):
    def __init__(self, m):
        self.content = m.content
        o = m.owner
        self.owner = o.name
        self.owner_name = o.display_name
        self.date = m.date_created

class Bug(Base):
    def __init__(self, bt):
        # Cache a few things for which launchpad will make a web request each
        # time.
        bug = bt.bug
        o = bt.owner
        a = bt.assignee
        dupe = bug.duplicate_of
        # Store from the launchpadlib bug objects only what we want, and as
        # local data
        self.id = bug.id
        self.lp_url = 'https://bugs.launchpad.net/%s/+bug/%i' % \
                      (PROJECT_NAME, self.id)
        self.title = bt.title
        self.description = bug.description
        # Every bug has an owner (who created it)
        self.owner = o.name
        self.owner_name = o.display_name
        # Not all bugs have been assigned to someone yet
        try:
            self.assignee = a.name
            self.assignee_name = a.display_name
        except AttributeError:
            self.assignee = self.assignee_name = None
        # Store status/importance in canonical format
        self.status = canonical_status(bt.status)
        self.importance = canonical_importance(bt.importance)
        self.tags = bug.tags
        # Store the bug discussion messages, but skip m[0], which is the same
        # as the bug description we already stored
        self.messages = map(Message, list(bug.messages)[1:])
        self.milestone = getattr(bt.milestone, 'name', None)

        # Duplicate handling disabled, since the default query already filters
        # out the duplicates.  Keep the code here in case we ever want to look
        # into this...
        if 0:
            # Track duplicates conveniently
            try:
                self.duplicate_of = dupe.id
                self.is_duplicate = True
            except AttributeError:
                self.duplicate_of = None
                self.is_duplicate = False

            # dbg dupe info
            if bug.number_of_duplicates > 0:
                self.duplicates = [b.id for b in bug.duplicates]
            else:
                self.duplicates = []

        # tmp - debug
        self._bt = bt
        self._bug = bug

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Launchpad part
#-----------------------------------------------------------------------------
# launchpad = create_session()
launchpad = Launchpad.login_with('statsmodels', 'production')
project = launchpad.projects[PROJECT_NAME]
# Note: by default, this will give us all bugs except duplicates and those
# with status "won't fix" or 'invalid'
bug_tasks = project.searchTasks(status=lp_status)

bugs = {}
for bt in list(bug_tasks):
    b = Bug(bt)
    bugs[b.id] = b
    print b.title
    sys.stdout.flush()

#-----------------------------------------------------------------------------
# Github part
#-----------------------------------------------------------------------------
#http://pypi.python.org/pypi/github2
#http://github.com/ask/python-github2
# Github libraries
from github2 import core, issues, client
for mod in (core, issues, client):
   reload(mod)


def format_title(bug):
    return bug.title.split('{0}: '.format(PROJECT_ID), 1)[1].strip('"')


def format_body(bug):
    body = \
"""Original Launchpad bug {bug.id}: {bug.lp_url}
Reported by: {bug.owner} ({owner_name}).

{description}""".format(bug=bug, owner_name=bug.owner_name.encode('utf-8'),
                    description=bug.description.encode('utf-8'))
    return body


def format_message(num, m):
    body = \
"""[ LP comment {num} by: {owner_name}, on {m.date!s} ]

{content}""".format(num=num, m=m, owner_name=m.owner_name.encode('utf-8'),
                    content=m.content.encode('utf-8'))
    return body


# Config
user = 'wesm'
token= '12efaff85b8e17f63ee835c5632b8cf0'

repo = 'statsmodels/statsmodels'
#repo = 'ipython/ipython'

# Skip bugs with this status:
# to_skip = set([u'fix_committed', u'incomplete'])
to_skip = set()

# Only label these importance levels:
gh_importances = set([u'critical', u'high', u'low', u'medium', u'wishlist'])

# Start script
gh = client.Github(username=user, api_token=token)

# Filter out the full LP bug dict to process only the ones we want
bugs_todo = dict( (id, b) for (id, b) in bugs.iteritems()
                  if not b.status in to_skip )

# Select which bug ids to run
#bids = bugs_todo.keys()[50:100]
# bids = bugs_todo.keys()[12:]

bids = bugs_todo.keys()
#bids = bids[:5]+[502787]

# Start loop over bug ids and file them on Github
nbugs = len(bids)
gh_issues = []  # for reporting at the end
for n, bug_id in enumerate(bids):
    bug = bugs[bug_id]
    title = format_title(bug)
    body = format_body(bug)

    print
    if len(title)<65:
        print bug.id, '[{0}/{1}]'.format(n+1, nbugs), title
    else:
        print bug.id, title[:65]+'...'

    # still check bug.status, in case we manually added other bugs to the list
    # above (mostly during testing)
    if bug.status in to_skip:
        print '--- Skipping - status:',bug.status
        continue

    print '+++ Filing...',
    sys.stdout.flush()

    # Create github issue for this bug
    issue = gh.issues.open(repo, title=title, body=body)
    print 'created GitHub #', issue.number
    gh_issues.append(issue.number)
    sys.stdout.flush()

    # Mark status as a label
    #status = 'status-{0}'.format(b.status)
    #gh.issues.add_label(repo, issue.number, status)

    # Mark any extra tags we might have as labels
    for tag in b.tags:
        label = 'tag-{0}'.format(tag)
        gh.issues.add_label(repo, issue.number, label)

    # If bug has assignee, add it as label
    if bug.assignee:
        gh.issues.add_label(repo, issue.number,
                            #bug.assignee
                            # Github bug, gets confused with dots in labels.
                            bug.assignee.replace('.','_')
                            )

    if bug.importance in gh_importances:
        if bug.importance == 'wishlist':
            label = bug.importance
        else:
            label = 'prio-{0}'.format(bug.importance)
        gh.issues.add_label(repo, issue.number, label)

    if bug.milestone:
        label = 'milestone-{0}'.format(bug.milestone).replace('.','_')
        gh.issues.add_label(repo, issue.number, label)

    # Add original message thread
    for num, message in enumerate(bug.messages):
        # Messages on LP are numbered from 1
        comment = format_message(num+1, message)
        gh.issues.comment(repo, issue.number, comment)
        time.sleep(0.5) # soft sleep after each message to prevent gh block

    if bug.status in ['fix_committed', 'fix_released', 'invalid']:
        gh.issues.close(repo, issue.number)

    # too many fast requests and gh will block us, so sleep for a while
    # I just eyeballed these values by trial and error.
    time.sleep(1) # soft sleep after each request
    # And longer one after every batch
    batch_size = 10
    tsleep = 60
    if (len(gh_issues) % batch_size)==0:
        print
        print '*** SLEEPING for {0} seconds to avoid github blocking... ***'.format(tsleep)
        sys.stdout.flush()
        time.sleep(tsleep)

# Summary report
print
print '*'*80
print 'Summary of GitHub issues filed:'
print gh_issues
print 'Total:', len(gh_issues)
