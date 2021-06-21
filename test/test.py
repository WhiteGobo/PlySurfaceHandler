import unittest
import itertools
import sys
import importlib
import os.path
testpath, tmp = os.path.split( __file__ )
mainpath, tmp = os.path.split( testpath )
asdf = os.path.join( mainpath, "__init__.py")
parentpath, modulename = os.path.split( mainpath )
_main_loader = importlib.machinery.SourceFileLoader( modulename, asdf )
main = _main_loader.load_module()

import tempfile


class test_asdf( unittest.TestCase ):
    def test_load_ply( self ):
        tmpfile = os.path.join( testpath, "multiplesurface.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        testarray = ((-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), \
                (-1, 0.5, 0), (-1, 0, 0), (-1, -0.5, 0), (-0.5, -1, 0), \
                (0, -1, 0), (0.5, -1, 0), (1, -0.5, 0), (1, 0, 0), \
                (1, 0.5, 0), (0.5, 1, 0), (0, 1, 0), (-0.5, 1, 0), \
                (-0.5, -0.5, 0), (-0.5, 0, 0), (-0.5, 0.5, 0), (0, -0.5, 0), \
                (0, 0, 0), (0, .5, 0), (.5, -.5, 0), (.5, 0, 0), (.5, .5, 0) )
        self.assertEqual( tuple( asdf.get_vertexpositions() ), testarray )
        testarray = ( (24,12,3,13), (4,18,15,2), (18,21,14,15), (21,24,13,14), \
                (0, 7, 16, 6), (6, 16, 17, 5), (5, 17, 18, 4), (7, 8, 19, 16), \
                (16,19,20,17), (17,20,21,18), (8,9,22,19), (19, 22, 23, 20), \
                (20,23,24,21), (9,1,10,22), (22,10,11,23), (23, 11, 12, 24) )
        self.assertEqual( tuple(asdf.get_faceindices()), testarray )
        self.assertEqual( asdf.get_number_surfaces(), 2 )
        surf1 = asdf.get_surface( 0 )
        for a, b in zip((7, 0, 2, 15), (surf1.rightup, surf1.leftup, \
                                        surf1.leftdown, surf1.rightdown)):
            self.assertEqual( a, b )
        surf2 = asdf.get_surface( 1 )
        for a, b in zip((9, 13, 3, 1), (surf2.rightup, surf2.leftup, \
                                        surf2.leftdown, surf2.rightdown)):
            self.assertEqual( a, b )
        self.assertEqual( "surf1", surf1.surfacename )
        self.assertEqual( "surf2", surf2.surfacename )
        self.assertEqual( tuple(range(25)), tuple(surf1.vertexlist) )
        self.assertEqual( tuple(range(25)), tuple(surf2.vertexlist) )

    def test_save_ply_easyfacesvertices( self ):
        vertices = ((-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0))
        faces = ( (0,1,2), (1,2,3) )
        vertices = [ main.vertex( *v ) for v in vertices ]
        faces = [ main.face( f ) for f in faces ]
        qwer = main.plysurfacehandler( vertices, faces )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join( tmpdir, "tmpfile.ply" )
            qwer.save_to_file( filepath )
            asdf = main.plysurfacehandler.load_from_file( filepath )
            self.assertEqual( tuple( asdf.get_vertexpositions() ), \
                               tuple( qwer.get_vertexpositions() ) )

    def test_save_ply_simplesurface( self ):
        vertices = ((-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0))
        faces = ( (0,1,2), (1,2,3) )
        surface = main.surface( rightup=3, leftup=2, leftdown=0, \
                                    rightdown=1, \
                                    surfacename="asdf", \
                                    vertexlist=(0,1,2,3), faceindices=faces)
        vertices = [ main.vertex( *v ) for v in vertices ]
        faces = [ main.face( f ) for f in faces ]
        qwer = main.plysurfacehandler( vertices, faces, (surface,) )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join( tmpdir, "tmpfile.ply" )
            qwer.save_to_file( filepath )
            asdf = main.plysurfacehandler.load_from_file( filepath )
            self.assertEqual( tuple( asdf.get_vertexpositions() ), \
                                tuple( qwer.get_vertexpositions() ) )
            get_corn = lambda x: tuple(( x.rightup, x.leftup, \
                                        x.leftdown, x.rightdown ))
            self.assertEqual( get_corn( asdf.get_surface(0) ), \
                                get_corn( surface ) )
            self.assertEqual( asdf.get_surface(0).surfacename, \
                                qwer.get_surface(0).surfacename )
            self.assertEqual( tuple(asdf.get_surface(0).vertexlist), \
                                tuple(qwer.get_surface(0).vertexlist) )

    def test_create_gridmap( self ):
        surfut = importlib.import_module( "plysurfacehandler.surfacemap_utils" )
        tmpfile = os.path.join( testpath, "singlesurface.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        vertexpositions = asdf.get_vertexpositions()
        faces = asdf.get_faceindices()
        edges = ( itertools.chain( *(zip( f[:], f[1:]+f[:1] ) for f in faces )))
        edges = set( frozenset( e ) for e in edges )
        edges = list( tuple(e) for e in edges )

        surf = asdf.get_surface( 0 )
        up, left, down, right = surf.get_borders()
        vertexpositions = list( vertexpositions )
        surfmap = surfut.create_gridmap_from( \
                                                    up, left, down, right, \
                                                    vertexpositions, edges )
        surfmap.visualize_with_matplotlib()



if __name__ == "__main__":
    unittest.main()
