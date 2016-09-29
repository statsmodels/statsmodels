#!/usr/bin/env python
"""
This script installs the trunk version, builds the docs, then uploads them
to ...

Then it installs the devel version, builds the docs, and uploads them to
...

Depends
-------
virtualenv

Notes
-----
If you set it up as an anacron job, you should do it as a user, so that it
has access to your ssh keys.
"""
import traceback
import base64
import subprocess
import os
import re
import shutil
import smtplib
import sys
from urllib2 import urlopen
from email.MIMEText import MIMEText
import logging
logging.basicConfig(filename='/home/skipper/statsmodels/statsmodels/tools/'
                             'docs_build_config.log', level=logging.DEBUG,
                    format="%(asctime)s %(message)s")
sys.stdout = open('/home/skipper/statsmodels/statsmodels/tools/crontab.out',
                  'w')
sys.stderr = open('/home/skipper/statsmodels/statsmodels/tools/crontab.err',
                  'w')

# Environment for subprocess calls. Needed for cron execution
env = {'MATPLOTLIBRC' : ('/home/skipper/statsmodels/statsmodels/tools/'),
       'HOME' : '/home/skipper',
       'PATH' : ':'.join((os.getenv('PATH', ''), '/home/skipper/.local/bin')),
       # Need this for my openblas setup on my laptop
       # maybe no longer necessary with newer numpy
       'LD_LIBRARY_PATH' : os.getenv('LD_LIBRARY_PATH', '')}


######### INITIAL SETUP ##########

#hard-coded "current working directory" ie., you will need file permissions
#for this folder
# follow symbolic links
script = os.path.realpath(sys.argv[0])
dname = os.path.abspath(os.path.dirname(script))
gname = 'statsmodels'
gitdname = os.path.join(dname, gname)
os.chdir(dname)

logging.debug('script: {}'.format(script))
logging.debug('dname: {}'.format(dname))

# sourceforge account for rsync
sf_account = 'jseabold,statsmodels@web.sourceforge.net'

# hard-coded git branch names
repo = 'git://github.com/statsmodels/statsmodels.git'
stable_trunk = 'master'
last_release = 'v0.5.0'
branches = [stable_trunk]
# change last_release above and uncomment the below to update for a release
#branches = [stable_trunk, last_release]

# virtual environment directory
virtual_dir = 'BUILDENV'
virtual_dir = os.path.join(dname, virtual_dir)
# this points to the newly installed python in the virtualenv
virtual_python = os.path.join(virtual_dir, 'bin', 'python')

# my security holes
with open('/home/skipper/statsmodels/gmail.txt') as f:
    pwd = f.readline().strip()
gmail_pwd = base64.b64decode(pwd)

########### EMAIL #############
email_name = 'statsmodels.dev' + 'AT' + 'gmail' + '.com'
email_name = email_name.replace('AT', '@')
gmail_pwd = gmail_pwd
to_email = [email_name,
            ('josef.pktd' + 'AT' + 'gmail' + '.com').replace('AT', '@')]
#to_email = [email_name]


########### FUNCTIONS ###############

def create_virtualenv():
    # make a virtualenv for installation if it doesn't exist
    # and easy_install sphinx
    if not os.path.exists(virtual_dir):
        retcode = subprocess.call(['/home/skipper/.local/bin/virtualenv',
                                   "--system-site-packages", virtual_dir],
                                   stderr=sys.stderr, stdout=sys.stdout)
        if retcode != 0:
            msg = """There was a problem creating the virtualenv"""
            raise Exception(msg)
        retcode = subprocess.call([virtual_dir+'/bin/easy_install', 'sphinx'])
        if retcode != 0:
            msg = """There was a problem installing sphinx"""
            raise Exception(msg)


def create_update_gitdir():
    """
    Creates a directory for local repo if it doesn't exist,
    updates repo otherwise.
    """
    if not os.path.exists(gitdname):
        retcode = subprocess.call('git clone '+repo, shell=True,
                                  stdout=sys.stdout, stderr=sys.stderr)
        if retcode != 0:
            msg = """There was a problem cloning the repo"""
            raise Exception(msg)
    else:  # directory exists, can't pull if you're not on a branch
           # just delete it and clone again. Lazy but clean solution.
        shutil.rmtree(gitdname)
        create_update_gitdir()


def check_version(branch, latest_hash=None):
    if branch == 'master':
        remote_dir = 'devel'
        regex = ("(?<=This documentation is for version <b>\d{1}\.\d{1}\."
                 "\d{1}\.dev-)(\w{7})")
    else:
        remote_dir = 'stable'
        regex = ("(?<=This documentation is for the <b>)(\d{1}\.\d{1}\.\d{1})"
                 "(?=</b> release.)")
    base_url = 'http://www.statsmodels.org/{}'
    page = urlopen(base_url.format(remote_dir)).read()

    try:
        version = re.search(regex, page).group()
    except AttributeError:
        return True

    if remote_dir == 'stable':
        if last_release[1:] == version:
            return False
        else:
            return True

    # get the lastest hash
    if latest_hash == version:
        return False
    else:
        return True


def getdirs():
    """
    Get current directories of cwd in order to restore to this
    """
    dirs = [i for i in os.listdir(dname)]
    dirs = filter(lambda x : not os.path.isfile(os.path.join(dname, x)),
                  dirs)
    return dirs


def newdir(dirs):
    """
    Returns difference in directories between dirs and current directories

    If the difference is greater than one directory it raises an error.
    """
    dirs = set(dirs)
    newdirs = set([i for i in os.listdir(dname) if not
                   os.path.isfile(os.path.join(dname, i))])
    newdir = newdirs.difference(dirs)
    if len(newdir) != 1:
        msg = ("There was more than one directory created. Don't know what "
               "to delete.")
        raise Exception(msg)
    newdir = newdir.pop()
    return newdir


def install_branch(branch):
    """
    Installs the branch in a virtualenv.
    """
    # if it's already in the virtualenv, remove it
    ver = '.'.join(map(str, (sys.version_info.major, sys.version_info.minor)))
    sitepack = os.path.join(virtual_dir, 'lib', 'python'+ver, 'site-packages')
    if os.path.exists(sitepack):
        dir_list = os.listdir(sitepack)
    else:
        dir_list = []
    for f in dir_list:
        if 'statsmodels' in f:
            shutil.rmtree(os.path.join(sitepack, f))

    # checkout the branch
    os.chdir(gitdname)
    retcode = subprocess.call('git checkout ' + branch, shell=True,
                              stdout=sys.stdout, stderr=sys.stderr)
    if retcode != 0:
        msg = """Could not checkout out branch %s""" % branch
        raise Exception(msg)

    p = subprocess.Popen('git rev-parse HEAD ', shell=True,
                              stdout=subprocess.PIPE, stderr=sys.stderr)
    version = p.communicate()[0][:7]

    # build and install
    retcode = subprocess.call(" ".join([virtual_python, 'setup.py', 'build']),
                              shell=True, stdout=sys.stdout, stderr=sys.stderr)
    if retcode != 0:
        msg = """Could not build branch %s""" % branch
        raise Exception(msg)
    retcode = subprocess.call(" ".join([virtual_python, os.path.join(gitdname,
                                        'setup.py'), 'install']), shell=True,
                              stdout=sys.stdout, stderr=sys.stderr)
    if retcode != 0:
        os.chdir(dname)
        msg = """Could not install branch %s""" % branch
        raise Exception(msg)
    os.chdir(dname)
    return version


def print_info():
    subprocess.Popen([virtual_python, os.path.join(gitdname,
                                                   "statsmodels",
                                                   "tools",
                                                   "print_version.py"
                                                   )],
                     stdout=sys.stdout, stderr=sys.stderr)


def build_docs(branch):
    """
    Changes into gitdname and builds the docs using BUILDENV virtualenv
    """
    os.chdir(os.path.join(gitdname, 'docs'))
    retcode = subprocess.call("make clean", shell=True,
                              stdout=sys.stdout, stderr=sys.stderr)
    if retcode != 0:
        os.chdir(dname)
        msg = """Could not clean the html docs for branch %s""" % branch
        raise Exception(msg)
    #NOTE: The python call in the below makes sure that it uses the Python
    # that is referenced after entering the virtualenv
    sphinx_call = " ".join(['make', 'html', "SPHINXBUILD=' python "
                            "/usr/local/bin/sphinx-build'"])
    activate = os.path.join(virtual_dir, "bin", "activate")
    activate_virtualenv = ". " + activate
    #NOTE: You have to enter virtualenv in the same call. As soon as the
    # child process is done, the env variables from activate are lost.
    # getting the correct env from bin/activate and passing to env is
    # annoying
    retcode = subprocess.call(" && ".join([activate_virtualenv, sphinx_call]),
                              shell=True, env=env, stdout=sys.stdout,
                              stderr=sys.stderr)

    if retcode != 0:
        os.chdir(dname)
        msg = """Could not build the html docs for branch %s""" % branch
        raise Exception(msg)
    os.chdir(dname)


def build_pdf(branch):
    """
    Changes into new_branch_dir and builds the docs using sphinx in the
    BUILDENV virtualenv
    """
    os.chdir(os.path.join(gitdname, 'statsmodels', 'docs'))
    sphinx_dir = os.path.join(virtual_dir, 'bin')
    retcode = subprocess.call(" ".join(['make', 'latexpdf',
                              'SPHINXBUILD='+sphinx_dir+'/sphinx-build']),
                              shell=True)
    if retcode != 0:
        msg = """Could not build the pdf docs for branch %s""" % branch
        raise Exception(msg)
    os.chdir(dname)


def upload_docs(branch):
    if branch == 'master':
        remote_dir = 'devel'
    else:
        remote_dir = 'stable'
    os.chdir(os.path.join(gitdname, 'docs'))
    retcode = subprocess.call(['rsync', '-avPzh', '--inplace', '-e ssh',
                               'build/html/', sf_account + ':htdocs/' +
                               remote_dir],
                              stderr=sys.stderr, stdout=sys.stdout)
    if retcode != 0:
        msg = """Could not upload html to %s for branch %s""" % (remote_dir,
                                                                 branch)
        raise Exception(msg)
    os.chdir(dname)


#TODO: upload pdf is not tested
def upload_pdf(branch):
    if branch == 'master':
        remote_dir = 'devel'
    else:
        remote_dir = 'stable'
    os.chdir(os.path.join(dname, new_branch_dir, 'statsmodels','docs'))
    retcode = subprocess.call(['rsync', '-avP', '-e ssh',
                               'build/latex/statsmodels.pdf',
                               sf_account + ':htdocs/' + remote_dir + 'pdf/'])
    if retcode != 0:
        msg = ("Could not upload pdf to %s for branch %s" %
               (remote_dir+'/pdf', branch))
        raise Exception(msg)
    os.chdir(dname)


def email_me(status='ok'):
    if status == 'ok':
        message = """
    HTML Documentation uploaded successfully.
    """
        subject = "Statsmodels HTML Build OK"
    else:
        message = status
        subject = "Statsmodels HTML Build Failed"

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = email_name
    msg['To'] = email_name

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(email_name, gmail_pwd)
    server.sendmail(email_name, to_email, msg.as_string())
    server.close()


############### MAIN ###################

def main():
    # get branch, install in virtualenv, build the docs, upload, and cleanup
    msg = ''
    for branch in branches:
        try:
            create_virtualenv()
            create_update_gitdir()
            version = install_branch(branch)
            if check_version(branch, version):
                print_info()
                build_docs(branch)
                upload_docs(branch)
            else:
                msg += ('Latest version already available for branch '
                        '{}.\n'.format(branch))
    #        build_pdf(new_branch_dir)
    #        upload_pdf(branch, new_branch_dir)
        except:
            msg += traceback.format_exc()

    if msg == '':  # if it doesn't something went wrong and was caught above
        email_me()
    else:
        email_me(msg)

if __name__ == "__main__":
    main()
