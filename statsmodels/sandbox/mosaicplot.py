from __future__ import division

import numpy
import pylab
from numpy import iterable,r_,cumsum,array
from collections import OrderedDict
from itertools import product
from pylab import Rectangle,figure
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def normalize_split(proportion):
    """return a list of proportions of the available space given the division
    if only a number is given, it will assume a split in two pieces"""
    if not iterable(proportion):
        if proportion==0:
            proportion = array([0.0,1.0])
        elif proportion>=1:
            proportion = array([1.0,0.0])
        elif proportion<0:
            raise ValueError('proportions should be positive, given value: {}'.format(proportion))
        else:
            proportion = array([proportion,1.-proportion])
    proportion = numpy.asarray(proportion,dtype=float)
    if numpy.any(proportion<0):
        raise ValueError('proportions should be positive, given value: {}'.format(proportion))
    if numpy.allclose(proportion,0):
        raise ValueError('at least one proportion should be greater than zero'.format(proportion))
    #ok, data are meaningful, so go on
    if len(proportion)<2:
        return array([0.0,1.0])
    left = r_[0,cumsum(proportion)]
    left /= left[-1]*1.
    return left

def split_rect(x,y,width,height,proportion,horizontal=True,gap=0.05):
    """split the given rectangle in n segments whose proportion is specified along the given axis
    if a gap is inserted, they will be separated by a certain amount of space, retaining the relative proportion between them
    a gap of 1 correspond to a plot that is half void and the remaining half space is proportionally divided
    among the pieces"""
    #take everything as a float
    x,y,w,h = float(x),float(y),float(width),float(height)
    if (w<0) or (h<0):
        raise ValueError("dimension of the square less than zero w={} h=()".format(w,h))
    proportions = normalize_split(proportion)
    #extract the starting point and the dimension of each subdivision
    #in respect to the unit square
    starting = proportions[:-1]
    amplitude = proportions[1:] - starting
    # how much each extrema is going to be displaced due to gaps
    starting += gap * numpy.arange(len(proportions)-1)
    #how much the squares plus the gaps are extended
    extension = starting[-1]+amplitude[-1]-starting[0]
    #normalize everything for fit again in the original dimension
    starting/=extension
    amplitude/=extension
    
    #bring everything to the original square
    starting = (x if horizontal else y) + starting * ( w if horizontal else h )
    amplitude = amplitude * ( w if horizontal else h )
    
    #create each 4-tuple for each new block
    results = [ (s,y,a,h) if horizontal else (x,s,w,a) for s,a in zip(starting,amplitude) ]
    return results


def reduce_dict(count_dict,partial_key):
    """make partial sum on a counter dict.
    given a match for the beginning of the category, it will sum each value"""
    L = len(partial_key)
    count = sum( v for k,v in count_dict.iteritems() if k[:L]==partial_key )
    return count

def key_splitting(rect_dict,keys,values,key_subset,horizontal,gap):
    """given a dictionary where each entry  is a rectangle, a list of key and value (count of elements in each category)
    it split each rect accordingly, as long as the key start with the tuple key_subset.
    the other keys are returned without modification"""
    result = OrderedDict()
    L = len(key_subset)
    for name, (x,y,w,h) in rect_dict.iteritems():
        if key_subset == name[:L]:
            #split base on the values given 
            divisions = split_rect(x,y,w,h,values,horizontal,gap)
            for key,rect in zip(keys,divisions):
                result[name+(key,)] = rect
        else:
            result[name] = (x,y,w,h)
    return result

def recursive_splitting(count_dict,horizontal=True,gap=0.05):
    base_rect = OrderedDict([ (tuple(),(0,0,1,1)) ])
    # use the Ordered dict to implement a simple ordered set
    #return each level of each category
    # [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]
    categories_levels = [ list(OrderedDict([(j,None) for j in i])) for i in zip(*(count_dict.keys())) ]
    # put the count dictionay in order for the keys
    # thil will allow some code simplification
    count_ordered = OrderedDict( [ (k,count_dict[k]) for k in list(product(*categories_levels)) ] )
    for cat_idx,cat_enum in enumerate(categories_levels):
        #get the partial key up to the actual level
        base_keys = list(product(*categories_levels[:cat_idx]))
        for key in base_keys:
            #for each partial and each value calculate how many observation we have in the
            #counting dictionary
            part_count= [ reduce_dict(count_ordered,key+(partial,)) for partial in cat_enum ]
            #reduce the gap for subsequents levels
            new_gap = gap/1.5**cat_idx
            #split the given subkeys in the rectangle dictionary
            base_rect = key_splitting(base_rect,cat_enum,part_count,key,horizontal,new_gap)
        horizontal = not horizontal
    return base_rect

def coord2rect(rect_dict):
    """from the dictionary of coordinates create the rectangles patches"""
    return OrderedDict([ [k,((x,y,w,h),Rectangle((x,y),w,h))] for k,(x,y,w,h) in rect_dict.items() ])

single_rgb_to_hsv=lambda rgb: rgb_to_hsv( array(rgb).reshape(1,1,3) ).reshape(3)
single_hsv_to_rgb=lambda hsv: hsv_to_rgb( array(hsv).reshape(1,1,3) ).reshape(3)


def mosaic(data,ax=None,horizontal=True,gap=0.005,decorator=None,labelizer=None):
    """
    it create the actual plot:
        takes the set of boxes of the division with the ticks
        use the decorator to generate the patches
        draw the patches
        draw the appropriate ticks on the plot
    """
    if ax is None:
        ax=pylab.gca()
    
    res = recursive_splitting(data,horizontal=horizontal,gap=gap)
    rects = coord2rect(res)
    
    if decorator is None:
        categories = [ list(OrderedDict([(j,None) for j in i])) for i in zip(*(data.keys())) ]
        L = [1.*len(cat) for cat in categories]
        props = [ numpy.linspace(0,1,l+2)[1:-1] for l in L ]
        if len(L)==4:
            props[3]=[ '', 'x', '/', '\\', '|', '-', '+',  'o', 'O', '.', '*' ]
        def dec(cat,rect):
            prop = [ props[k][categories[k].index(cat[k])] for k in range(len(cat)) ]
            hsv = [0., 0.4, 0.7]
            for idx,i in enumerate(prop[:3]):
                hsv[idx]=i
            hatch = prop[3] if len(prop)==4 else ''
            rect.set_color(single_hsv_to_rgb(hsv))
            rect.set_hatch(hatch)
        decorator = dec
    
    if labelizer is None:
        labelizer =lambda k: "\n".join(k)+"\ncount="+str(data[k]) 
    
    for k,(v,r) in rects.items():
        x,y,w,h = v
        test = labelizer(k)
        r.set_label(test)
        decorator(k,r)
        ax.add_patch(r)
        ax.text(x+w/2,y+h/2,test,ha='center',va='center')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])  
    
    #for idx,t in enumerate(ticks):
    #    for (lab,pos) in zip(*t):
    #        s = 0.02
    #        border= -s if idx<2 else 1+s
    #        valign= 'top' if idx<2 else 'baseline'
    #        halign= 'right' if idx<2 else 'left'
    #        x,y,v,h = (border,pos,'center',halign) if (direction =='v')!=(not idx%2) else (pos,border,valign,'center')
    #        size = ['xx-large','x-large','large','large','medium','medium','small','x-small'][idx]
    #        ax.text(x,y,lab,horizontalalignment = h, verticalalignment = v,size=size,rotation=0)
    return rects





