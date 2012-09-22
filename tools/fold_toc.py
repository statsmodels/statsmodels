#!/usr/bin/env python
import sys
import re

# Read doc to string
filename = sys.argv[1]
doc = open(filename).read()

# Add mktree to head
pre = '<head>'
post = '''<head>
    <script type="text/javascript" src="_static/mktree.js"></script>
    <link rel="stylesheet" href="_static/mktree.css" type="text/css">
'''
doc = re.sub(pre, post, doc)

# TOC class 
pre = '''<div class="toctree-wrapper compound">
<ul>'''
post = '''<div class="toctree-wrapper compound">
Click <tt>+</tt> to expand and <tt>-</tt> to collapse.  
<a onclick="collapseTree('toctree')" href="javascript:void(0);">Collapse all. </a>
<a onclick="expandTree('toctree')" href="javascript:void(0);">Expand all. </a>
<ul class="mktree" id="toctree">'''
doc = re.sub(pre, post, doc)

## TOC entries
pre = '<li class="toctree-l1">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 2nd level
pre = '<li class="toctree-l2">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 3rd level
pre = '<li class="toctree-l3">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 4th level
pre = '<li class="toctree-l4">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# Write to file
f = open(filename, 'w')
f.write(doc)
f.close()

