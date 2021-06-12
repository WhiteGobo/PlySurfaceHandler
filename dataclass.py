from .exceptions import DatacontainerLoadError
from .constants import FORMAT_X, FORMAT_Y, FORMAT_Z
from .plyhandler import ObjectSpec as PlyObject

class plysurfacehandler():
    def __init__( self, vertexlist= None, facelist=None, surfacelist=None ):
        self.vertices = copy.deepcopy( vertexlist )
        self.facelist = copy.deepcopy( facelist )
        self.surfacelist = copy.deepcopy( surfacelist )

    def check_valid( self ):
        vertexiterator = iter( self.vertexdata )
        firstvertex = vertexiterator.__next__()
        for v in vertexiterator:
            if not vertexiterator.is_compatible_to( v ):
                return False

    def load_vertexdata_array( self, data, dataformat="auto"):
        if dataformat=="auto":
            dataformat = self._automatic_dataformat( data )

    def load_vertexdata( self, x=None, y=None, z=None, ):
        if self.vertexdata is not None:
            for d in (x,y,z):
                if d is not None:
                    if len(d) != len(self.vertexdata):
                        raise DatacontainerLoadError()
        if x is not None:
            for vertex, tmpx in zip( self.vertexdata, x ):
                vertex.x = tmpx
        if y is not None:
            for vertex, tmpy in zip( self.vertexdata, y ):
                vertex.y = tmpy
        if z is not None:
            for vertex, tmpz in zip( self.vertexdata, z ):
                vertex.z = tmpz

    @classmethod
    def load_from_file( cls, filepath ):
        plyobj = PlyObject.load_from_file( filepath )
        vertexdata, number_vertices = _get_vertexdata( plyobj )
        facedata, number_faces = _get_facedata( plyobj )
        surfacedata, number_surfaces = _get_surfacedata( plyobj )
        vertices, faces, surfaces = [], [], []
        for i in range( number_vertices ):
            tmp = vertex( **{ key: arg[i] for key, arg in vertexdata.items() } )
            vertices.append( tmp )
        for i in range( number_faces ):
            tmp = face( **{ key:arg[i] for key, arg in facedata.items() } )
            faces.append( tmp )
        for i in range( number_surfaces ):
            tmp = surface( **{ key: arg[i] for key, arg in surfacedata.items()})
            surfaces.append( tmp )
        return cls( vertices, faces, surfaces )

    def _automatic_dataformat( self, data ):
        return "".join((FORMAT_X, FORMAT_Y, FORMAT_Z))

class surface():
    def __init__( self, rightup=None, leftup=None, leftdown=None, \
                                    rightdown=None, vertexlist=None ):
        self.rightup = rightup
        self.leftup = rightup
        self.leftdown = rightup
        self.rightdown = rightdown

class face():
    def __init__( self, vertexindices=None ):
        self.vertexindices = vertexindices

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
    return faceindices, number_faces

def _get_surfacedata( plyobj ):
    border = plyobj.get_filtered_data( "cornerrectangle",\
                                            ("rightup", "leftup", \
                                            "leftdown", "rightdown") )
    return borderdata, number_surfaces

def asdf():
    ply_name = bpy.path.display_name_from_filepath( filepath )
    meshname = ply_name
    objectname = ply_name

    #vertexlist, faces, rightup, leftup, leftdown, rightdown \
    vertexlist, faces, borders, bordernames = load_meshdata_from_ply( filepath )


def load_meshdata_from_ply( filepath ):
    """
    :todo: use f cr is shitty
    """
    plyobj = PlyObject.load_from_file( filepath )
    try:
        vertexpositions = plyobj.get_filtered_data("vertex", ("x", "y", "z") )
        faceindices = plyobj.get_filtered_data( "face", ("vertex_indices",) )
        faceindices = [ f[0] for f in faceindices ]
        border = plyobj.get_filtered_data( "cornerrectangle",\
                                            ("rightup", "leftup", \
                                            "leftdown", "rightdown") )
    except KeyError as err:
        raise InvalidPlyDataForSurfaceobject( "couldnt find all needed "\
                        "elements and associated properties that are needed" )\
                        from err

    bordernames = (None,)
    try:
        bordernames = plyobj.get_filtered_data("cornerrectangle", ("surfacename",))
        bordernames = [ "".join(chr(i) for i in name[0]) \
                        for name in bordernames ]
    except KeyError:
        pass
    return vertexpositions, faceindices, border, bordernames
