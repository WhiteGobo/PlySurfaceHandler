import unittest
import sys
import importlib
import os.path
testpath, tmp = os.path.split( __file__ )
mainpath, tmp = os.path.split( testpath )
asdf = os.path.join( mainpath, "__init__.py")
parentpath, modulename = os.path.split( mainpath )
_main_loader = importlib.machinery.SourceFileLoader( modulename, asdf )
main = _main_loader.load_module()


class test_asdf( unittest.TestCase ):
    def test_load_ply( self ):
        tmpfile = os.path.join( testpath, "multiplesurface.ply" )
        asdf = main.plysurfacehandler.load_from_file( tmpfile )
        pass
    def test_save_ply( self ):
        pass


if __name__ == "__main__":
    unittest.main()
