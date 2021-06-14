from .exceptions import DatacontainerLoadError
from .constants import FORMAT_X, FORMAT_Y, FORMAT_Z
from .plyhandler import ObjectSpec as PlyObject
import copy
from typing import Iterator, Generator

class face():
    def __init__( self, vertex_indices=None ):
        self.vertex_indices = tuple( vertex_indices )

class vertex():
    def __init__( self, x=None, y=None, z=None ):
        self.x = x
        self.y = y
        self.z = z
    def is_compatible_to( self, other_vertex):
        conditions = (
                (self.x is not None and other_vertex.x is not None) \
                or (self.x is None and other_vertex.x is None), \
                (self.y is not None and other_vertex.y is not None) \
                or (self.y is None and other_vertex.y is None), \
                (self.z is not None and other_vertex.z is not None) \
                or (self.z is None and other_vertex.z is None),
                )
        return all( conditions )
class surface():
    def __init__( self, rightup=None, leftup=None, leftdown=None, \
                                    rightdown=None, surfacename=None, \
                                    vertexlist:Iterator[int]=None, \
                                    faceindices:Iterator[Iterator[int]] =None):
        clist = (rightup, leftup, leftdown, rightdown)
        if not any( (all(c is None for c in clist), \
                    all(c is not None for c in clist)) ):
            raise TypeError( "rightup, leftup, leftdown or rightdown is wrong" )
        self.rightup = int(rightup)
        self.leftup = int(leftup)
        self.leftdown = int(leftdown)
        self.rightdown = int(rightdown)
        self.surfacename = surfacename
        self.vertexlist = vertexlist
        if self.vertexlist is not None:
            try:
                vertex_trans, used_faces = self._create_translator( vertexlist,\
                                                                faceindices )
            except TypeError as err:
                raise TypeError( "vertexlist or faceindices has wrong type" ) from err
            if faceindices is None:
                used_faces = None
        else:
            vertex_trans, used_faces = None, None
        self.vertex_trans = vertex_trans
        self.used_faces = used_faces

    def _create_translator( self, vertexmask, faceindices ):
        vertex_trans = { a:i for i, a in enumerate( sorted(vertexmask) ) }
        used_faces = [ i for i, face in enumerate(faceindices) \
                        if all( index in vertexmask for index in face ) ]
        return vertex_trans, used_faces

class plysurfacehandler():
    def __init__( self, vertexlist: Iterator[ vertex ] = None, \
                    facelist: Iterator[ face ] = None, \
                    surfacelist: Iterator[ surface ] = None ):
        self._facelist = copy.deepcopy( facelist )
        self._surfacelist = copy.deepcopy( surfacelist )
        self._vertices = copy.deepcopy( vertexlist )
        if vertexlist is not None:
            assert type( vertexlist[0] ) == vertex, "plysurfhandler worng input"
        if facelist is not None:
            assert type( facelist[0] ) == face, "plysurfhandler worng input"
        if surfacelist is not None:
            assert type( surfacelist[0] ) ==surface,"plysurfhandler worng input"

    def get_vertexpositions( self ) -> Iterator[ Iterator[float] ]:
        """
        Returns the vertices as positiontuples, eg: [(0,1,2),(3,4,5)]
        automaticly recognizes how many coordinates there are (1,2 or 3)
        in case of only one coordinate available list contains instead of
        tuples the coordinates directly, eg: [0,3]
        """
        if self._vertices[0].z is not None:
            for v in self._vertices:
                yield (v.x, v.y, v.z)
        elif self._vertices[0].y is not None:
            for v in self._vertices:
                yield (v.x, v.y)
        else:
            for v in self._vertices:
                yield v.x

    def get_faceindices( self ) -> Iterator[ Iterator[int] ]:
        for f in self._facelist:
            yield f.vertex_indices

    def get_number_surfaces( self ) -> int:
        return len( self._surfacelist ) if self._surfacelist is not None else 0

    def get_surface( self, index:int ) -> Iterator[ surface ]:
        surf = self._surfacelist[ index ]
        return surf

    def check_valid( self ):
        """
        tests if data is not corrupted
        :todo: complete this function
        """
        vertexiterator = iter( self.vertexdata )
        firstvertex = vertexiterator.__next__()
        for v in vertexiterator:
            if not vertexiterator.is_compatible_to( v ):
                return False

    #def load_vertexdata( self, x=None, y=None, z=None, ):
    #    if self.vertexdata is not None:
    #        for d in (x,y,z):
    #            if d is not None:
    #                if len(d) != len(self.vertexdata):
    #                    raise DatacontainerLoadError()
    #    if x is not None:
    #        for vertex, tmpx in zip( self.vertexdata, x ):
    #            vertex.x = tmpx
    #    if y is not None:
    #        for vertex, tmpy in zip( self.vertexdata, y ):
    #            vertex.y = tmpy
    #    if z is not None:
    #        for vertex, tmpz in zip( self.vertexdata, z ):
    #            vertex.z = tmpz

    def save_to_file( self, filename:str, use_ascii=True, use_bigendian=False ):
        vertexinfo = self._create_vertexinfo_for_plyobject()
        faceinfo = self._create_faceinfo_for_plyobject()
        partialsurfaceinfo = self._create_surfaceinfo_for_plyobject()
        allinfo = [ d for d in (vertexinfo, faceinfo, partialsurfaceinfo) \
                    if d is not None ]
        myobj = PlyObject.from_arrays( allinfo )

        #theoreticly "binary_big_endian" is also possible
        if use_ascii:
            myformat = "ascii"
        elif use_bigendian:
            myformat = "binary_big_endian"
        else:
            myformat = "binary_little_endian" 
        myobj.save_to_file( filename, myformat )

    def _create_vertexinfo_for_plyobject( self ):
        vertices = list( self.get_vertexpositions() )
        mycheck = lambda x: x.is_integer() if type(x)==float else type(x)==int
        all_ints = all( all( mycheck(i) for i in v) for v in vertices )
        vertices = tuple( tuple( v[i] for v in vertices ) for i in range(3) )
        if all_ints:
            vertexpipeline = ( ( "int", "x" ), ( "int", "y" ), ( "int","z"))
        else:
            vertexpipeline = ( ("float", "x" ), ("float", "y" ), ("float","z"))
        return ("vertex", vertexpipeline, vertices )

    def _create_faceinfo_for_plyobject( self ):
        face_indices = list( self.get_faceindices() )
        facespipeline = ( ("list", "uchar", "uint", "vertex_indices" ), )
        return ("face", facespipeline, (face_indices,) )

    def _create_surfaceinfo_for_plyobject( self ):
        number = self.get_number_surfaces()
        if number == 0:
            return None
        surf = self.get_surface( 0 )
        borderpipeline = []
        borderdata = []
        if surf.surfacename is not None:
            borderpipeline.append( ("list", "uchar", "uchar", "surfacename" ) )
            sn = [ bytes(self.get_surface(i).surfacename, encoding="utf8") \
                    for i in range( number ) ]
            borderdata.append( sn )
        if surf.rightup is not None:
            borderpipeline.extend([ ("uint", "rightup"), ("uint", "leftup"), \
                                ("uint", "leftdown"), ("uint", "rightdown") ])
            ru, lu, ld, rd = [], [], [], []
            borderdata.extend( (ru, lu, ld, rd) )
            for i in range( number ):
                surf = self.get_surface( i )
                ru.append( surf.rightup )
                lu.append( surf.leftup )
                ld.append( surf.leftdown )
                rd.append( surf.rightdown )
        if surf.vertexlist is not None:
            if len( list(self.get_vertexpositions()) ) < 255:
                borderpipeline.append(( "list", "uchar", "uchar", "vertexlist"))
            else:
                borderpipeline.append(( "list", "uint", "uint", "vertexlist" ))
            sn = [ self.get_surface(i).vertexlist for i in range( number ) ]
            borderdata.append( sn )
        return ("cornerrectangle", tuple(borderpipeline), tuple(borderdata) )

        #for surf in self._surfacelist:
        #    borderindices = list( 
        #        np.array( cornerdata ).T.reshape((4, len(surfacenames))) )
        if surfacenames != (None,):
            borderpipeline.append( ("list", "uchar", "uchar", "surfacename" ) )
            sn = [ bytes(name, encoding="utf8") for name in surfacenames ]
            borderindices.append( sn )

    @classmethod
    def load_from_file( cls, filepath:str ):
        plyobj = PlyObject.load_from_file( filepath )
        vertexdata, number_vertices = _get_vertexdata( plyobj )
        facedata, number_faces = _get_facedata( plyobj )
        surfacedata, number_surfaces = _get_surfacedata( plyobj )
        vertices = [] if number_vertices != 0 else None
        faces = [] if number_faces != 0 else None
        surfaces = [] if number_surfaces != 0 else None
        for i in range( number_vertices ):
            tmp = vertex( **{ key: arg[i] for key, arg in vertexdata.items() } )
            vertices.append( tmp )
        for i in range( number_faces ):
            tmp = face( **{ key:arg[i] for key, arg in facedata.items() } )
            faces.append( tmp )
        if number_surfaces > 0:
            faceindices = tuple(( face.vertex_indices for face in faces ))
        for i in range( number_surfaces ):
            tmp = surface( **{ key: arg[i] \
                                        for key, arg in surfacedata.items()},\
                                        faceindices = faceindices )
            surfaces.append( tmp )
        return cls( vertices, faces, surfaces )


def _get_vertexdata( plyobj ):
    vertexdata = {}
    #vertexpositions = plyobj.get_filtered_data("vertex", ("x", "y", "z") )
    number_vertices = plyobj.get_length_element( "vertex" )
    for dataname in ( 'x', 'y', 'z' ):
        vertexdata[ dataname ] = plyobj.get_dataarray( "vertex", dataname )
    return vertexdata, number_vertices

def _get_facedata( plyobj ):
    faceindices = plyobj.get_filtered_data( "face", ("vertex_indices",) )
    faceindices = [ f[0] for f in faceindices ]
    facedata = { "vertex_indices": faceindices }
    number_faces = len( faceindices )
    return facedata, number_faces

def _get_surfacedata( plyobj ):
    borderdata = {}
    number_surfaces = plyobj.get_length_element( "cornerrectangle" )
    if number_surfaces == 0:
        return None, number_surfaces
    for dataname in ( "rightup", "leftup", "leftdown", "rightdown", \
                        "surfacename", "vertexlist" ):
        try:
            borderdata[ dataname ] = plyobj.get_dataarray( "cornerrectangle", \
                                                                    dataname )
        except KeyError:
            pass


    if "surfacename" in borderdata:
        borderdata[ "surfacename" ] = [ "".join(chr(i) for i in name) \
                                    for name in borderdata[ "surfacename" ]]
    return borderdata, number_surfaces
