#!/usr/bin/python
"""
This script installs the trunk version, builds the docs, then uploads them
to ...

Then it installs the devel version, builds the docs, and uploads them to
...

Depends
-------
virtualenv
"""
import base64
import subprocess
import os
import shutil
import re
import smtplib
import sys
from email.MIMEText import MIMEText

######### INITIAL SETUP ##########

#hard-coded "curren working directory" ie., you will need file permissions
#for this folder
script = os.path.abspath(sys.argv[0])
dname = os.path.abspath(os.path.dirname(script))
gname = 'statsmodels'
gitdname = os.path.join(dname, gname)
os.chdir(dname)

# hard-coded git branch names
repo = 'git://github.com/statsmodels/statsmodels.git'
stable_trunk = 'master'
last_release = 'v0.3.1'
#branches = [stable_trunk, last_release]
#NOTE: just update the releases by hand
branches = [stable_trunk]

# virtual environment directory
virtual_dir = 'BUILDENV'
virtual_dir = os.path.join(dname, virtual_dir)
# this points to the newly installed python in the virtualenv
virtual_python = os.path.join(virtual_dir,'bin','python')


# my security holes
with open('/home/skipper/statsmodels/gmail.txt') as f:
    pwd = f.readline().strip()
gmail_pwd = base64.b64decode(pwd)

########### EMAIL #############
email_name ='statsmodels.dev' + 'AT' + 'gmail' +'.com'
email_name = email_name.replace('AT','@')
gmail_pwd= gmail_pwd
to_email = [email_name, ('josef.pktd' + 'AT' + 'gmail' + '.com').replace('AT',
'@')]


########### FUNCTIONS ###############

def create_virtualenv():
    # make a virtualenv for installation if it doesn't exist
    # and easy_install sphinx
    if not os.path.exists(virtual_dir):
        retcode = subprocess.call(['/usr/local/bin/virtualenv', virtual_dir])
        if retcode != 0:
            msg = """There was a problem creating the virtualenv"""
            raise Exception(msg)
        retcode = subprocess.call([virtual_dir+'/bin/easy_install', 'sphinx'])
        if retcode != 0:
            msg = """There was a problem installing sphinx"""
            raise Exception(msg)

def create_update_gitdir():
    """
    Creates a directory for local repo if it doesn't exist, updates repo otherwise.
    """
    if not os.path.exists(gitdname):
        retcode = subprocess.call('git clone '+repo, shell=True)
        if retcode != 0:
            msg = """There was a problem cloning the repo"""
            raise Exception(msg)
    else:
        os.chdir(gitdname)
        retcode = subprocess.call('git pull', shell=True)
        if retcode != 0:
            msg = """There was a problem pulling from the repo."""
            raise Exception(msg)

def getdirs():
    """
    Get current directories of cwd in order to restore to this
    """
    dirs = [i for i in os.listdir(dname) if not \
            os.path.isfile(os.path.join(dname, i))]
    return dirs

def newdir(dirs):
    """
    Returns difference in directories between dirs and current directories

    If the difference is greater than one directory it raises an error.
    """
    dirs = set(dirs)
    newdirs = set([i for i in os.listdir(dname) if not \
            os.path.isfile(os.path.join(dname,i))])
    newdir = newdirs.difference(dirs)
    if len(newdir) != 1:
        msg = """There was more than one directory created.  Don't know what to delete."""
        raise Exception(msg)
    newdir = newdir.pop()
    return newdir

def install_branch(branch):
    """
    Installs the branch in a virtualenv.
    """

    # if it's already in the virtualenv, remove it
    ver = '.'.join(map(str,(sys.version_info.major,sys.version_info.minor)))
    sitepack = os.path.join(virtual_dir,'lib','python'+ver, 'site-packages')
    dir_list = os.listdir(sitepack)
    for f in dir_list:
        if 'scikits.statsmodels' in f:
            shutil.rmtree(os.path.join(sitepack, f))

    # checkout the branch
    os.chdir(gitdname)
    retcode = subprocess.call('git checkout ' + branch, shell=True)
    if retcode != 0:
        msg = """Could not checkout out branch %s""" % branch
        raise Exception(msg)

    # build and install
    retcode = subprocess.call(" ".join([virtual_python, 'setup.py', 'build']),
                                shell=True)
    if retcode != 0:
        msg = """ Could not build branch %s""" % branch
        raise Exception(msg)
    retcode = subprocess.call(" ".join([virtual_python, os.path.join(gitdname,
        'setup.py'), 'install']), shell=True)
    if retcode != 0:
        os.chdir(dname)
        msg = """Could not install branch %s""" % branch
        raise Exception(msg)
    os.chdir(dname)

def build_docs(branch):
    """
    Changes into gitdname and builds the docs using sphinx in the
    BUILDENV virtualenv
    """
    os.chdir(os.path.join(gitdname,'scikits','statsmodels','docs'))
    sphinx_dir = os.path.join(virtual_dir,'bin')
    #NOTE: don't use make.py, just use make and specify which sphinx
    #    retcode = subprocess.call([virtual_python,'make.py','html',
    #        '--sphinx_dir='+sphinx_dir])
    retcode = subprocess.call(" ".join(['make','html',
        'SPHINXBUILD='+sphinx_dir+'/sphinx-build']), shell=True)
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
    os.chdir(os.path.join(gitdname,'scikits','statsmodels','docs'))
    sphinx_dir = os.path.join(virtual_dir,'bin')
    retcode = subprocess.call(" ".join(['make','latexpdf',
        'SPHINXBUILD='+sphinx_dir+'/sphinx-build']), shell=True)
    if retcode != 0:
        os.chdir(old_cwd)
        msg = """Could not build the pdf docs for branch %s""" % branch
        raise Exception(msg)
    os.chdir(dname)

def upload_docs(branch):
    if branch == 'master':
        remote_dir = 'devel'
    else:
        remote_dir = ''
    #    old_cwd = os.getcwd()
    os.chdir(os.path.join(gitdname,'scikits','statsmodels','docs'))
    retcode = subprocess.call(['rsync', '-avPr' ,'-e ssh', 'build/html/',
        'jseabold,statsmodels@web.sourceforge.net:htdocs/'+remote_dir])
    if retcode != 0:
        os.chdir(old_cwd)
        msg = """Could not upload html to %s for branch %s""" % (remote_dir, branch)
        raise Exception(msg)
    os.chdir(dname)

#TODO: upload pdf is not tested
def upload_pdf(branch):
    if branch == 'master':
        remote_dir = 'devel'
    else:
        remote_dir = ''
    os.chdir(os.path.join(dname, new_branch_dir,'scikits','statsmodels','docs'))
    retcode = subprocess.call(['rsync', '-avPr', '-e ssh',
        'build/latex/statsmodels.pdf',
        'jseabold,statsmodels@web.sourceforge.net:htdocs/'+remote_dir+'pdf/'])
    if retcode != 0:
        os.chdir(old_cwd)
        msg = """Could not upload pdf to %s for branch %s""" % (remote_dir+'/pdf',
                branch)
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

    server = smtplib.SMTP('smtp.gmail.com',587)
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
            #create virtualenv
            create_virtualenv()
            create_update_gitdir()
            install_branch(branch)
            build_docs(branch)
            upload_docs(branch)
    #        build_pdf(new_branch_dir)
    #        upload_pdf(branch, new_branch_dir)
        except Exception as status:
            msg += status.args[0] + '\n'

    if msg == '': # if it doesn't something went wrong and was caught above
        email_me()
    else:
        email_me(msg)

if __name__ == "__main__":
    main()
