Star98 Educational Dataset
==========================

Description
-----------


This data is on the California education policy and outcomes (STAR program
results for 1998.  The data measured standardized testing by the California
Department of Education that required evaluation of 2nd - 11th grade students
by the the Stanford 9 test on a variety of subjects.  This dataset is at
the level of the unified school district and consists of 303 cases.  The
binary response variable represents the number of 9th graders scoring
over the national median value on the mathematics exam.

The data used in this example is only a subset of the original source.


Notes
-----

Number of Observations - 303 (counties in California).

Number of Variables - 13 and 8 interaction terms.

Definition of variables names::

    NABOVE   - Total number of students above the national median for the math
               section.
    NBELOW   - Total number of students below the national median for the math
               section.
    LOWINC   - Percentage of low income students
    PERASIAN - Percentage of Asian student
    PERBLACK - Percentage of black students
    PERHISP  - Percentage of Hispanic students
    PERMINTE - Percentage of minority teachers
    AVYRSEXP - Sum of teachers' years in educational service divided by the
               number of teachers.
    AVSALK   - Total salary budget including benefits divided by the number of
               full-time teachers (in thousands)
    PERSPENK - Per-pupil spending (in thousands)
    PTRATIO  - Pupil-teacher ratio.
    PCTAF    - Percentage of students taking UC/CSU prep courses
    PCTCHRT  - Percentage of charter schools
    PCTYRRND - Percentage of year-round schools

    The below variables are interaction terms of the variables defined above.

    PERMINTE_AVYRSEXP
    PEMINTE_AVSAL
    AVYRSEXP_AVSAL
    PERSPEN_PTRATIO
    PERSPEN_PCTAF
    PTRATIO_PCTAF
    PERMINTE_AVTRSEXP_AVSAL
    PERSPEN_PTRATIO_PCTAF


Source
------

Jeff Gill's `Generalized Linear Models: A Unified Approach`

http://jgill.wustl.edu/research/books.html


Copyright
---------

Used with express permission from the original author,
who retains all rights.
