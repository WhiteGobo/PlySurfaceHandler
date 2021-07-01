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
import numpy as np


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
        #surfut =importlib.import_module( "plysurfacehandler.surfacemap_utils" )
        tmpfile = os.path.join( testpath, "singlesurface.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        surfmap = asdf.create_surfacemap( 0 )
        #surfmap.visualize_with_matplotlib()

    def test_create_gridmap_second( self ):
        #surfut =importlib.import_module( "plysurfacehandler.surfacemap_utils" )
        tmpfile = os.path.join( testpath, "singlesurfacesecond.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        surfmap = asdf.create_surfacemap( 0 )
        #surfmap.visualize_with_matplotlib()

    def test_complete_and_save_gridmaps_little( self ):
        tmpfile = os.path.join( testpath, "singlesurface.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        asdf.complete_surfaces_with_map()
        matr1 = np.array( asdf.get_surface(0).get_datamatrix_of_surfacematrix())
        matr2 = np.array( asdf.get_surface(0).get_surfacemap().datamatrix)
        self.assertTrue( np.allclose( matr1, matr2 ))
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join( tmpdir, "tmpfile.ply" )
            asdf.save_to_file( filepath )
            qwer = main.plysurfacehandler.load_from_file( filepath )
            matr3 = qwer.get_surface(0).get_datamatrix_of_surfacematrix()
            self.assertIsNotNone( matr3 )
            self.assertTrue( np.allclose( matr1, np.array(matr3) ))

    def test_complete_and_save_gridmaps_big( self ):
        tmpfile = os.path.join( testpath, "testbig.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        asdf.complete_surfaces_with_map()
        matr1 = np.array( asdf.get_surface(0).get_datamatrix_of_surfacematrix())
        matr2 = np.array( asdf.get_surface(0).get_surfacemap().datamatrix)
        self.assertTrue( np.allclose( matr1, matr2 ))
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join( tmpdir, "tmpfile.ply" )
            asdf.save_to_file( filepath )
            qwer = main.plysurfacehandler.load_from_file( filepath )
            matr3 = qwer.get_surface(0).get_datamatrix_of_surfacematrix()
            self.assertIsNotNone( matr3 )
            self.assertTrue( np.allclose( matr1, np.array(matr3) ))
        #asdf.get_surface(0).get_surfacemap().visualize_with_matplotlib()

    def test_get_lengths( self ):
        tmpfile = os.path.join( testpath, "singlesurface_with_map.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        matr3 = asdf.get_surface(0).get_datamatrix_of_surfacematrix()
        mysurfmap = asdf.get_surface(0).get_surfacemap()

        testarray = [0.0, 0.07142857, 0.1428571, 0.2142857, 0.2857143, 0.3571429, 0.4285714, 0.5, 0.5714286, 0.6428571, 0.7142857, 0.7857143, 0.8571429, 0.9285714, 1.0]
        self.assertTrue( np.allclose( testarray, mysurfmap.uplength ) )
        self.assertTrue( np.allclose( testarray, mysurfmap.leftlength ) )
        self.assertTrue( np.allclose( testarray, mysurfmap.downlength ) )
        self.assertTrue( np.allclose( testarray, mysurfmap.rightlength ) )




if __name__ == "__main__":
    unittest.main()
