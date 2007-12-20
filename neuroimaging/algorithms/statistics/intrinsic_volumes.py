import os
from scipy.weave import ext_tools
import numpy as N
import string

"""
The estimators for the intrinsic volumes appearing in this module
were partially supported by NSF grant DMS-0405970.

Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
   with an application to brain mapping."
   Journal of the American Statistical Association, 102(479):913-928.

"""

ntab = 4
space = " "*ntab
tab = space

def _intrep(x):
    """
    Return integer value based on a binary representation
    """

    return (N.array([2**i for i in range(N.asarray(x).shape[0])][::-1]) * x).sum()

def _binrep(x, dim=3):
    """
    Return binary representation of x (assumed less than 2**dim)
    """
    x = int(x)
    if x >= 2**dim:
        raise ValueError, 'x too large, increase dim'
    a = []
    while x > 0:
        x, bit = (x / 2), x % 2
        a.append(bit)
    a = a[::-1]
    return [0]*(dim-len(a)) + a

def subsets(n, k=None, fixed=None, empty=False):
    """
    All subsets of range(n), returned as binary
    vectors of length n, with options:

    fixed: if fixed is not None, it returns only those subsets with
           all indexes in fixed excluded

    k: if not None, then filter for size k

    empty: if True, return the emptyset

    """

    keep = [_binrep(i, dim=n) for i in range(2**n)]
    if not empty:
        keep = keep[1:]
    if fixed is not None:
        keep = filter(lambda v: N.product([v[i] == 0 for i in fixed]), keep)
    if k is None:
        return keep
    else:
        return filter(lambda x: N.sum(x) == k, keep)


class Complex:

    """
    A simple class that takes a list of maximal simplices (by
    default a triangulation of a cube into 6 tetrahedra) and
    computes all faces, edges, vertices.
    """

    def __init__(self, maximal=[(0, 3, 2, 7),
                                (0, 6, 2, 7),
                                (0, 7, 5, 4),
                                (0, 7, 5, 1),
                                (0, 7, 4, 6),
                                (0, 3, 1, 7)]):

        self.faces = {}

        l = [len(list(x)) for x in maximal]
        for i in range(N.max(l)):
            self.faces[i+1] = set([])

        for simplex in maximal:
            simplex = list(simplex)
            simplex.sort()
            for k in range(1,len(simplex)+1):
                ss = subsets(len(simplex), k=k)
                for s in ss:
                    v = tuple([simplex[i] for i in N.nonzero(s)[0]])
                    if len(v) == 1:
                        v = v[0]
                    self.faces[k].add(v)

cube = Complex()

class Vertex:

    arrayname = 'mask'

    def __init__(self, ivector):
        self.vector = ivector
        self.dim = N.array(self.vector).shape[0]

    def nonzero(self):
        """
        [1,0,1] -> [0,2]
        """
        return list(N.nonzero(self.vector)[0])

    def zero(self):
        """
        [1,0,1] -> [1]
        """
        nz = set(self.nonzero())
        return list(set(range(self.dim)).difference(nz))

    def tostring(self):
        """
        [1,0,1] -> "101"
        """
        return string.join(["%d" % v for v in self.vector], '')

    def stride(self):
        nz = self.nonzero()
        if nz:
            if self.arrayname == 'mask':
                return "\n%slong n%s = (" % (space ,self.tostring()) + string.join(["Smask[%d]" % i for i in nz], ' + ') + ") / sizeof(*mask);"
            elif self.arrayname == 'resid':
                return "\n%slong nd%s = (" % (space ,self.tostring()) + string.join(["Sresid[%d]" % (i+1,) for i in nz], ' + ') + ") / sizeof(*resid);"
        else:
            if self.arrayname == 'mask':
                return "\n%slong n%s = 0;" % (space, self.tostring())
            elif self.arrayname == 'resid':
                return "\n%slong nd%s = 0;" % (space, self.tostring())

    def indicator(self):
        return 'm%s' % self.tostring()

    def declare(self, lk=0):
        value = "\n%sint m%s;" % (space, self.tostring())
        return value

class Face(Vertex):

    """
    A face of a hypercube.

    The vector determines which face of the cube it is, i.e.
    [1,0,1] indicates that the cube has the 1st and 3rd dimension changing.
    By default, the vertex [1,1,1] is always included, which determines
    the face uniquely.

    """

    def declare(self, lk=0):
        value = "\n%slong M%s;" % (space, self.tostring())
        if lk in [1,2,3]:
            value += "\n%sdouble lk%s;" % (space, self.tostring())
        return value

    def __init__(self, vector):
        Vertex.__init__(self, vector)
        self.dimface = N.array(self.vector).sum()
        if self.dim in [1,2,3]:
            self._get_cube()

    def _get_cube(self):
        nz = self.nonzero()
        if nz:
            self.dimface = N.array(nz).shape[0]
            vectors = N.zeros((self.dimface, self.dim), N.int)
            for i in range(self.dimface):
                vectors[i][nz[i]] = 1
            ivector = [_intrep(v) for v in vectors] + [-1] * (3-self.dimface)

            self.cube = {1:Complex(maximal=[(0,ivector[0])]),
                         2:Complex(maximal=[(0,ivector[0],ivector[1]),
                                            (ivector[0],ivector[1],ivector[0]+ivector[1])]),
                         3:Complex()}[self.dimface]

    def indicator(self, indiv=False, group=False):
        """
        [1,0,1] -> "m101*m001*m100"
        """
        value = ''
        K = len(self.nonzero())

        if indiv:
            for vv in self.cube.faces[1]:
                if vv != 0:
                    vv = Vertex(_binrep(vv, dim=self.dim))
                    value += space + "m%s = mask[n-n%s];\n" % (vv.tostring(), vv.tostring())

        if group:
            ovalue = []
            for v in self.cube.faces[1]:
                ovalue.append(Vertex(_binrep(v, dim=self.dim)).indicator())
            value += space + 'M%s = ' % self.tostring() + string.join(ovalue, '*') + ';\n'
        return value

    def where(self, lk=0):
        value = ''
        K = len(self.nonzero())
        nz = self.nonzero()
        value += tab*K + space + 'n = (' + string.join(["i%d * Smask[%d]" % (l, nz[l]) for l in range(K)], " + ") + ') / sizeof(*mask);\n'
        if lk in [1,2,3]:
            value += tab*K + space + 'nd = (' + string.join(["i%d * Sresid[%d]" % (l, nz[l]+1) for l in range(K)], " + ") + ') / sizeof(*resid);'
        return value

    def openfor(self):
        nz = self.nonzero()
        value = ''
        for k in range(len(nz)):
            value += string.join([tab*k + space + "for(i%d=1; i%d<Nmask[%d]; i%d++) {\n" % (k, k, nz[k], k)])
        return value

    def closefor(self):
        nz = self.nonzero()
        value = ''
        for k in range(len(nz)):
            value += tab*(len(nz)-1-k) + space + "}\n"
        return value

    def openif(self):
        K = len(self.nonzero())
        v = Vertex([0]*self.dim)
        value = """
%(sp)sm%(v)s = mask[n-n%(v)s];
%(sp)sif (m%(v)s) {
""" % {'sp':tab*K + space,
       'v':v.tostring()}
        return value

    def closeif(self):
        K = len(self.nonzero())
        return tab*K + space + '}\n'

    def intvolcode(self, lk=1):
        dimface = N.array(self.vector).sum()
        value = space + 'lk%s = 0;\n' % self.tostring()
        nz = self.nonzero()
        for i in range(lk+1, len(nz)+2):
            sign = {-1:'-',
                    1:'+'}[(-1)**(i+1-lk)]
            whatmap = {4:'tet',
                       3:'tri',
                       2:'edge'}
            for f in self.cube.faces[i]:
                value += space + "lk%s %s= mu%d_%s(resid, " % (self.tostring(),
                                                               sign,
                                                               lk,
                                                               whatmap[i])
                for j in range(i):
                    value += "nd-nd%s, " % Vertex(_binrep(f[j], dim=self.dim)).tostring()
                value += "Nresid[0], Sresid[0]);\n"
        return value

def code(dim, lk=1):

    explorer = False

    if explorer and dim==3:
        code = explorer_header
    else:
        code = ''

    if lk in [1,2,3]:
        code += "\n%slong nd, n;" % space
    else:
        code += "\n%slong n;" % space

    code += """
    %(space)sdouble LK = 0;%(space)sint %(i)s;\n""" % {'space':'\n' + space,
                                                       'i':string.join(["i%d" % i for i in range(dim)], ', ')}

    cubefaces = [_binrep(i, dim=dim) for i in range(2**dim)]

    for v in cubefaces:
        V = Vertex(v)
        code += V.declare()
        code += V.stride()
        if lk in [1,2,3]:
            V.arrayname = 'resid'
            code += V.stride()
        code += '\n'

    for f in cubefaces:
        F = Face(f)
        code += F.declare(lk=lk) + '\n'

    for v in cubefaces:
        F = Face(v)
        code += F.openfor()
        K = len(F.nonzero())
        nz = F.nonzero()
        if hasattr(F, 'cube'):
            code += F.where(lk=lk)
            code += F.openif()
            code += F.indicator(indiv=True).replace(space, tab*(K+1)+space) + '\n'

            if lk == 0:
                code += tab*(K+1) + space + "LK += m%s;\n" % (Vertex([0]*dim).tostring())

            for subF in [Face(f) for f in subsets(F.dim, fixed=F.zero())]:
                if lk in [1,2,3]:
                    code += subF.intvolcode(lk=lk).replace(space, tab*(K+1)+space)
                    code += subF.indicator(group=True).replace(space, tab*(K+1)+space)
                    code += tab*(K+1)+space + "LK %s= M%s * lk%s;\n\n" % ({-1:'-',
                                                                           1:'+'}[(-1)**(lk-len(subF.nonzero()))],
                                                                          subF.tostring(),
                                                                          subF.tostring())
                else:
                    code += subF.indicator(group=True).replace(space, tab*(K+1)+space)
                    code += tab*(K+1)+space + "LK %s= M%s;\n\n" % ({-1:'-',
                                                                    1:'+'}[(-1)**(lk-len(subF.nonzero()))],
                                                                   subF.tostring())


            code += F.closeif()
        else:
            if lk == 0:
                code += "LK += mask[0];\n"

        code += F.closefor()


    return code


support_code = """

double mu3_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride);

double mu2_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride);

double mu1_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride);

double mu2_tri(double *data, long v0, long v1, long v2, long ndata,
               long dstride);

double mu1_tri(double *data, long v0, long v1, long v2, long ndata,
               long dstride);

double mu1_edge(double *data, long v0, long v1, long ndata,
		long dstride);

double mu3_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride) {

  long idata, i0, i1;
  double *d0, *d1, *dref;
  double XTX[3][3] = {{0,0,0},
		      {0,0,0},
		      {0,0,0}};
  double mu = 0;

  for (i0=0; i0<3; i0++) {
    for (i1=0; i1<=i0; i1++) {

      dref = (double *) ((char *)data + v3 * sizeof(*data));

      switch(i0) {
      case 0:
	d0 = (double *) ((char *)data + v0 * sizeof(*data));
	break;

      case 1:
	d0 = (double *) ((char *)data + v1 * sizeof(*data));
	break;

      case 2:
	d0 = (double *) ((char *)data + v2 * sizeof(*data));
	break;

      }

      switch(i1) {
      case 0:
	d1 = (double *) ((char *)data + v0 * sizeof(*data));
	break;

      case 1:
	d1 = (double *) ((char *)data + v1 * sizeof(*data));
	break;

      case 2:
	d1 = (double *) ((char *)data + v2 * sizeof(*data));
	break;

      }

      for (idata=0; idata<ndata; idata++) {
	XTX[i0][i1] += ((*d0) - (*dref)) * ((*d1) - (*dref));
	d0 = (double *) ((char *)d0 + dstride);
	d1 = (double *) ((char *)d1 + dstride);
	dref = (double *) ((char *)dref + dstride);
      }
      XTX[i1][i0] = XTX[i0][i1];
    }
  }

  mu = sqrt((XTX[0][0] * (XTX[1][1] * XTX[2][2] - XTX[1][2] * XTX[2][1]) -
	     XTX[0][1] * (XTX[1][0] * XTX[2][2] - XTX[2][0] * XTX[1][2]) +
	     XTX[0][2] * (XTX[1][0] * XTX[2][1] - XTX[1][1] * XTX[2][0]))) / 6.;

  return(mu);
}


double mu2_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride) {

  double mu=0;

  mu += mu2_tri(data, v0, v1, v2, ndata, dstride);
  mu += mu2_tri(data, v0, v1, v3, ndata, dstride);
  mu += mu2_tri(data, v0, v2, v3, ndata, dstride);
  mu += mu2_tri(data, v1, v2, v3, ndata, dstride);
  return(mu * 0.5);

}


double mu1_tet(double *data, long v0, long v1, long v2, long v3, long ndata,
               long dstride) {

  long idata, i0, i1, isubset;
  double *d0, *d1;
  double XTX[4][4] = {{0,0,0,0},
		      {0,0,0,0},
		      {0,0,0,0},
		      {0,0,0,0}};
  double A[3][3] = {{0,0,0},
		    {0,0,0},
		    {0,0,0}};

  double length;
  double norm_proj[2];
  double inner_prod_proj;

  double mu=0;
  double pi=3.1415926535897931;
  int s0, s1, t0, t1;
  int subsets[6][4] = {{0,1,2,3},
		       {0,2,1,3},
		       {0,3,1,2},
		       {1,2,0,3},
		       {1,3,0,2},
		       {2,3,0,1}};

  for (i0=0; i0<4; i0++) {
    for (i1=0; i1<=i0; i1++) {

      switch(i0) {
      case 0:
	d0 = (double *) ((char *)data + v0 * sizeof(*data));
	break;

      case 1:
	d0 = (double *) ((char *)data + v1 * sizeof(*data));
	break;

      case 2:
	d0 = (double *) ((char *)data + v2 * sizeof(*data));
	break;

      case 3:
	d0 = (double *) ((char *)data + v3 * sizeof(*data));
	break;
      }

      switch(i1) {
      case 0:
	d1 = (double *) ((char *)data + v0 * sizeof(*data));
	break;

      case 1:
	d1 = (double *) ((char *)data + v1 * sizeof(*data));
	break;

      case 2:
	d1 = (double *) ((char *)data + v2 * sizeof(*data));
	break;

      case 3:
	d1 = (double *) ((char *)data + v3 * sizeof(*data));
	break;
      }

      for (idata=0; idata<ndata; idata++) {
	XTX[i0][i1] += ((*d0) * (*d1));
	d0 = (double *) ((char *)d0 + dstride);
	d1 = (double *) ((char *)d1 + dstride);
      }
      XTX[i1][i0] = XTX[i0][i1];
    }
  }

  for (isubset=0; isubset<6; isubset++) {

    s0 = subsets[isubset][0]; s1 = subsets[isubset][1];
    t0 = subsets[isubset][2]; t1 = subsets[isubset][3];

    A[0][0] = XTX[s1][s1] - 2 * XTX[s1][s0] + XTX[s0][s0];
    A[1][1] = XTX[t0][t0] - 2 * XTX[t0][s0] + XTX[s0][s0];
    A[2][2] = XTX[t1][t1] - 2 * XTX[t1][s0] + XTX[s0][s0];

    A[0][1] = XTX[s1][t0] - XTX[t0][s0] - XTX[s1][s0] + XTX[s0][s0]; A[1][0] = A[0][1];
    A[0][2] = XTX[s1][t1] - XTX[t1][s0] - XTX[s1][s0] + XTX[s0][s0]; A[2][0] = A[0][2];
    A[1][2] = XTX[t0][t1] - XTX[t0][s0] - XTX[t1][s0] + XTX[s0][s0]; A[2][1] = A[1][2];

    length = sqrt(A[0][0]);

    norm_proj[0] = A[1][1] - A[0][1] * A[0][1] / A[0][0];
    norm_proj[1] = A[2][2] - A[0][2] * A[0][2] / A[0][0];
    inner_prod_proj = A[1][2] - A[0][1] * A[0][2] / A[0][0];

    mu += (pi - acos(inner_prod_proj / sqrt(norm_proj[0] * norm_proj[1]))) * length / (2 * pi);

  }

  return(mu);
}


double mu3_tri(double *data, long v0, long v1, long v2, long ndata,
               long dstride) {
  return(0);
}

double mu2_tri(double *data, long v0, long v1, long v2, long ndata,
               long dstride) {

  long idata, i0, i1;
  double *d0, *d1, *dref;
  double XTX[2][2] = {{0,0},
		      {0,0}};
  double mu=0;

  for (i0=0; i0<2; i0++) {
    for (i1=0; i1<=i0; i1++) {

      dref = (double *) ((char *)(&(data[0])) + v2 * sizeof(*data));

      switch(i0) {
      case 0:
	d0 = (double *) ((char *)(&(data[0])) + v0 * sizeof(*data));
	break;

      case 1:
	d0 = (double *) ((char *)(&(data[0])) + v1 * sizeof(*data));
	break;

      }

      switch(i1) {
      case 0:
	d1 = (double *) ((char *)data + v0 * sizeof(*data));
	break;

      case 1:
	d1 = (double *) ((char *)data + v1 * sizeof(*data));
	break;

      }

      for (idata=0; idata<ndata; idata++) {
	XTX[i0][i1] += ((*d0) - (*dref)) * ((*d1) - (*dref));
	d0 = (double *) ((char *)d0 + dstride);
	d1 = (double *) ((char *)d1 + dstride);
	dref = (double *) ((char *)dref + dstride);
      }
      XTX[i1][i0] = XTX[i0][i1];
    }
  }

  mu = sqrt((XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[0][1])) / 2.;
  return(mu);
}


double mu1_tri(double *data, long v0, long v1, long v2, long ndata,
               long dstride) {

  double mu=0;

  mu += mu1_edge(data, v0, v1, ndata, dstride);
  mu += mu1_edge(data, v0, v2, ndata, dstride);
  mu += mu1_edge(data, v1, v2, ndata, dstride);
  return(mu * 0.5);

}

double mu3_edge(double *data, long v0, long v1, long ndata,
		long dstride) {
  return(0);
}

double mu2_edge(double *data, long v0, long v1, long ndata,
		long dstride) {
  return(0);
}

double mu1_edge(double *data, long v0, long v1, long ndata,
		long dstride) {

  long idata;
  double *d0, *d1;
  double length=0;

  d0 = (double *) ((char *)data + v0 * sizeof(*data));
  d1 = (double *) ((char *)data + v1 * sizeof(*data));

  for (idata=0; idata<ndata; idata++) {
    length += ((*d0) - (*d1)) * ((*d0) - (*d1));
    d0 = (double *) ((char *)d0 + dstride);
    d1 = (double *) ((char *)d1 + dstride);
  }
  return(sqrt(length));
}

"""


##

##     explorer_header = """

## #include <c:/Explorer50/include/cx/cxLattice.api.h>
## #include <cx/cxParameter.api.h>
## #include <cx/DataAccess.h>
## #include <cx/DataTypes.h>
## #include <cx/ModuleCommand.h>
## #include <cx/MinMax.h>
## #include <stdlib.h>
## #include <sys/types.h>
## #include <string.h>
## #include <stdio.h>
## #include <cx/Typedefs.h>

## void LK%dfunc(long nDim, long *Nresid, float *bBox, float *data,
##             float *mask, float datathresh, float maskthresh, float fwhm,
## 	    cxParameter *outLK) {

##    long Sresid[4] = {Nresid[1]*Nresid[2]*Nresid[3]*sizeof(double),
##                      Nresid[2]*Nresid[3]*sizeof(double),
##                      Nresid[3]*sizeof(double),
##                      sizeof(double)};
##    long Nmask[3] = {Nresid[1], Nresid[2], Nresid[3]};
##    long Smask[3] = {Sresid[1], Sresid[2], Sresid[3]};
##    char stringoutLK[512];

## """ % lk

def setup_extension():

    mod = ext_tools.ext_module('_intrinsic_volumes', compiler='gcc')
    mod.customize.add_support_code(support_code)
    EC = {}
    for i in range(1,6):
        mask = N.zeros((5,)*i, N.int)
        _code = code(i, lk=0)
        _code += space + 'return_val = LK;\n'
        EC = ext_tools.ext_function("ECdim%d" % i, _code, ['mask'])
        mod.add_function(EC)
        if i in range(1,4):
            resid = N.zeros((10,)+(5,)*i)
            for lk in range(1,4):
                if lk <= i:
                    _code = code(i, lk=lk)
                    _code += space + 'return_val = LK;\n'
                    LK = ext_tools.ext_function("LK%ddim%d" % (lk, i), _code, ['mask', 'resid'])
                    mod.add_function(LK)

    return mod



try:
    import _intrinsic_volumes
except:
    mod = setup_extension()
    d = mod.setup_extension(location=os.path.dirname(__file__)).__dict__
    n = d['name']; del(d['name'])
    s = d['sources']; del(d['sources'])
    d['include_dirs'].append(N.get_include())
    extension = n, s, d
    mod.compile()
    import _intrinsic_volumes

def EC(X, thresh=0):
    m = N.greater(X, thresh).astype(N.int)
    f = {1: _intrinsic_volumes.ECdim1,
         2: _intrinsic_volumes.ECdim2,
         3: _intrinsic_volumes.ECdim3,
         4: _intrinsic_volumes.ECdim4,
         5: _intrinsic_volumes.ECdim5}[len(X.shape)]
    return f(m)

def LK(X, thresh, coords=None, lk=0):
    m = N.greater(X, thresh).astype(N.int)
    try:
        f = {(1,1):_intrinsic_volumes.LK1dim1,
             (2,1):_intrinsic_volumes.LK1dim2,
             (2,2):_intrinsic_volumes.LK2dim2,
             (3,1):_intrinsic_volumes.LK1dim3,
             (3,2):_intrinsic_volumes.LK2dim3,
             (3,3):_intrinsic_volumes.LK3dim3,
             (1,0):_intrinsic_volumes.ECdim1,
             (2,0):_intrinsic_volumes.ECdim2,
             (3,0):_intrinsic_volumes.ECdim3,
             (4,0):_intrinsic_volumes.ECdim4,
             (5,0):_intrinsic_volumes.ECdim5}[(len(X.shape), lk)]
    except KeyError:
        raise KeyError, 'cannot compute this intrinsic volume'
    if lk > 0:
        if coords is None:
            raise ValueError, "need coords to compute intrinsic volumes"
        return f(m, coords.astype(N.float))
    else:
        return f(m)

