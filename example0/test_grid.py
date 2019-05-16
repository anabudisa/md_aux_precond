import porepy as pp
import numpy as np
import sys; sys.path.insert(0, '/home/anci/Dropbox/new_porepy/porepy/src/porepy/')
sys.path.insert(0, '../src/')

from Hcurl3D import Hcurl


def make_mesh(file_name, mesh_size, plot=True):
    mesh_kwargs = {'mesh_size_frac': mesh_size,
                   'mesh_size_min': mesh_size,
                   'mesh_size_bound': mesh_size}

    # g = pp.StructuredTetrahedralGrid([1, 1, 1])
    # g.compute_geometry()

    # The fractures are specified by their vertices, stored in a numpy array
    f_1 = pp.Fracture(np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]]))
    f_2 = pp.Fracture(np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]]))

    # Also define the domain
    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 1}

    # Define a 3d FractureNetwork, similar to the 2d one
    network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
    mesh_args = {'mesh_size_frac': 1.75, 'mesh_size_min': 1.7}

    # Generate the mixed-dimensional mesh
    gb = network.mesh(mesh_kwargs, ensure_matching_face_cell=False)

    # make fracture network
    # network_3d = pp.fracture_importer.network_3d_from_csv(file_name)

    # Generate a mixed-dimensional mesh
    # gb = network_3d.mesh(mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()

    # grid_list = [g]

    # gb = pp.meshing.grid_list_to_grid_bucket(grid_list)

    hcurl = Hcurl(gb)
    hcurl.compute_edges()
    print(hcurl.num_edges)

    if plot:
        pp.plot_grid(gb, alpha=0, info="c")

    return gb


# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    # grid config file
    name = "network_3d.csv"

    make_mesh(name, 1.2)
