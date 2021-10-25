from typing import Iterable, Tuple
import itertools
from collections import Counter

class NoValidBorder( Exception ):
    pass

def get_border_from_faces( rightup:int, leftup:int, \
                            leftdown:int, rightdown:int, \
                            faces:Iterable[Iterable[int]] ) \
                            ->Tuple[ Iterable[int],...]:
    """

    :rtype: Tuple[ Tuple[int],Tuple[int],Tuple[int],Tuple[int]]
    """
    edges = ( itertools.chain( *(zip( f[:], f[1:]+f[:1] ) for f in faces )))
    edges = ( frozenset( e ) for e in edges )
    #edges = list( tuple(e) for e in edges )
    edgecount = Counter( edges )
    borderedges = ( e for e, count in edgecount.items() if count == 1 )
    neighbours = {}
    for vert1, vert2 in borderedges:
        neighbours.setdefault( vert1, list() ).append( vert2 )
        neighbours.setdefault( vert2, list() ).append( vert1 )
    visited = set( (rightup,) )
    borderindices = [ rightup ]
    lastnode = rightup
    for i in range( len(neighbours)-1 ):
        nextneighbours = neighbours[ lastnode ]
        try:
            lastnode = set( nextneighbours ).difference( visited ).pop()
        except KeyError as err:
            raise NoValidBorder( "no single continuous border" ) from err
        visited.add( lastnode )
        borderindices.append( lastnode )
    if not rightup in neighbours[ lastnode ]:
        raise NoValidBorder( "couldnt close border" )
    try:
        cornerindices = [ borderindices.index( c ) \
                            for c in (leftup, leftdown, rightdown) ]
    except ValueError as err:
        raise NoValidBorder( "cornerpoints arent in border" ) from err
    ci = cornerindices
    if not any( (ci[0] < ci[1] < ci[2], ci[0] > ci[1] > ci[2] ) ):
        raise NoValidBorder( "cornerpoints arent in the right order" )
    if ci[0] > ci[1] > ci[2]:
        borderindices.append( borderindices.pop(0) )
        borderindices.reverse()
        cornerindices = [ borderindices.index( c ) \
                            for c in (leftup, leftdown, rightdown) ]
    up = borderindices[ :cornerindices[0]+1 ]
    left = borderindices[ cornerindices[0]:cornerindices[1]+1 ]
    down = borderindices[ cornerindices[1]:cornerindices[2]+1 ]
    right = borderindices[ cornerindices[2]: ] + borderindices[:1]
    return up, left, down, right
