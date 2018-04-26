from DV.dv_img import dv_rgb
from pyhdf import SD

h5py_path1 = r'C:/Users/ninga/Downloads/MOD09A1.A2011241.h24v05.005.2011250062145.hdf'
img_name1 = '2011250062145.png'
h5py_path2 = r'C:/Users/ninga/Downloads/MOD09A1.A2011241.h25v05.005.2011250062212.hdf'
img_name2 = '2011250062212.png'


print('start')

hdf_obj = SD.SD(h5py_path1, SD.SDC.READ)
r = hdf_obj.select('sur_refl_b01')[:]
g = hdf_obj.select('sur_refl_b04')[:]
b = hdf_obj.select('sur_refl_b03')[:]
dv_rgb(r, g, b, img_name1, linear=2)
hdf_obj.end()

hdf_obj = SD.SD(h5py_path2, SD.SDC.READ)
r = hdf_obj.select('sur_refl_b01')[:]
g = hdf_obj.select('sur_refl_b04')[:]
b = hdf_obj.select('sur_refl_b03')[:]
dv_rgb(r, g, b, img_name2, linear=2)
hdf_obj.end()

print('success')
