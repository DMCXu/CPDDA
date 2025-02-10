import numpy as np
import os
from CPDDA import materials
from CPDDA import structures
from CPDDA import fields
from CPDDA import core
from CPDDA import post_processing
from CPDDA import visu

file_path = __file__
file_name = os.path.basename(file_path)

AR = 12  # Aspect ratio of particle kernels
d = 0.5  # Size of the cubic lattice
L_core = 80  # Length of the particle kernel
s = 5  # Thickness of pellet shell

'set material for struct'
# material1 is the shell of particle, material2 is the core of particle
wavelength = np.arange(650, 655, 5, dtype=np.float32)
material1 = materials.FromDatabase(shelf_name="main", book_name="Au", page_name="Johnson.yml", wl=wavelength, nb=1.44)
material2 = materials.FromDatabase(shelf_name="main", book_name="ZnO", page_name="Querry.yml", wl=wavelength, nb=1.44)

'set struct object'
material = [material1, material2]
geometry = structures.rod(AR, d, L_core, s)
occupied = structures.INDEX_in("rod", geometry, L_core, s, AR)
struct = structures.struct(d, material, occupied, geometry)
print(struct)

'set field object'
field_generator = fields.plan_wave()
K_kwargs = dict(K0=np.array([1, 0, 0]))
E_kwargs = dict(E0=np.array([0, 0, 1]))
efield = fields.efield(field_generator, wavelength, K_kwargs, E_kwargs)
print(efield)

'set simulation object and running'
sim = core.simulation(struct, efield, file_name)
print(sim.field.field_type[0])
px, py, pz, PX_vector, PY_vector, PZ_vector, Inverse_Alpha, data = sim.DDA(method="cupy")

'the post processing and visualization of result'
Aext, Aabs, Asca = post_processing.vol_coe(data, sim)
visu.visu_vol_coe(sim, Aext, Aabs, Asca)
