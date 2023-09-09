import ctypes
import os


bin_dll_path = os.path.join(os.getcwd(), "bin_module", "bin_module.dll")
DLL = ctypes.CDLL(bin_dll_path)

def bradley_binariation(pixels, width, height, bradley_param):
    """Takes list of grayskale pixels and converts to black-white accroding to Bradley algorihtm\n
    Args: pixels(List), width(int), height(int), bradley_param(int)
    Output: None
"""
    bradley_bin = ctypes.PYFUNCTYPE(None, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object)(('bradley_binarization', DLL))
    bradley_bin(pixels, width, height, bradley_param)


bradley_bin = ctypes.PYFUNCTYPE(None, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object)(('bradley_binarization', DLL))

bradley_bin.__doc__ = """Takes list of grayskale pixels and converts to black-white accroding to Bradley algorihtm\n
    Args: pixels(List), width(int), height(int), bradley_param(int)
    Output: None
"""
bradley_bin.__annotations__ = {
    'pixels':'grayscale pixels(list)',
    'width': 'image width(int)',
    'height': 'image height(int)',
    'bradley_param': 'number of secotors into which the image will be divided(int)'
    }

anti_aliasing = ctypes.PYFUNCTYPE(None, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object)(('anti_aliasing', DLL))
anti_aliasing.__doc__ = """Takes list of grayscale pixels and applies antialising\n
    Args: pixels(List), width(int), height(int), intensity(int)
    Output: None
"""
anti_aliasing.__annotations__ = {
    'pixels':'grayscale pixels(list)',
    'width': 'image width(int)',
    'height': 'image height(int)',
    'intensity': 'anti_aliasing intensity(int)'
    }
