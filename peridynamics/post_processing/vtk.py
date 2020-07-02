
import numpy as np

def write(fileName, title, coords, Damage, U):
    f = open(fileName, "w")

    f.write("# vtk DataFile Version 2.0\n")
    f.write("%s \n" % title)
    f.write("ASCII\n")
    f.write("\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    f.write("POINTS %s double\n" % int(len(Damage)))

    for i in range(0, len(coords[:, 0])):
        tmp = coords[i, :]
        f.write("{:f} {:f} {:f}\n".format(tmp[0], tmp[1], tmp[2]))

    f.write("\n")
    f.write("POINT_DATA %s\n" % int(len(Damage)))
    f.write("SCALARS damage double\n")
    f.write("LOOKUP_TABLE default\n")
    for i in range(0, len(Damage)):
        tmp = Damage[i]
        f.write("%f\n" % tmp)

    f.write("VECTORS displacements double \n")
    for i in range(0, len(U[:, 0])):
        tmp = U[i, :]
        f.write("{:f} {:f} {:f}\n".format(tmp[0], tmp[1], tmp[2]))

    f.close()

def writeNetwork(fileName, title, max_horizon_length, horizons_lengths, family, bond_stiffnesses_family, bond_critical_stretch_family):
    f = open(fileName, "w")

    f.write("# vtk DataFile Version 2.0\n")
    f.write("%s \n" % title)
    f.write("ASCII\n")
    f.write("\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    f.write("MAX_HORIZON_LENGTH %d \n" % int(max_horizon_length))
    f.write("NNODES %d \n" % int(len(horizons_lengths)))
# =============================================================================
#     f.write("HORIZONS\n")
#     for i in range(0, len(horizons[:, 0])):
#         tmp = horizons[i, :]
#         for j in range(0, max_horizon_length):
#             f.write("{:d} ".format(np.intc(tmp[j])))
#         f.write("\n")
#         
#     f.write("\n")
# =============================================================================
    f.write("HORIZONS_LENGTHS \n")
    for i in range(0, len(horizons_lengths)):
        tmp = np.intc(horizons_lengths[i])
        f.write("%d\n" % tmp)

    f.write("\n")
    
    f.write("FAMILY \n")
    for i in range(0, np.shape(family)[0]):
        tmp = family[i]
        for j in range(0, len(tmp)):
            f.write("{:d} ".format(np.intc(tmp[j])))
        f.write("\n")
        
    f.write("STIFFNESS \n")
    for i in range(0, np.shape(family)[0]):
        tmp = bond_stiffnesses_family[i]
        for j in range(0, len(tmp)):
            f.write("{:f} ".format(tmp[j]))
        f.write("\n")
    
    f.write("STRETCH \n")
    for i in range(0, np.shape(family)[0]):
        tmp = bond_critical_stretch_family[i]
        for j in range(0, len(tmp)):
            f.write("{:f} ".format(tmp[j]))
        f.write("\n")

    f.close()

def writeDamage(fileName, title, damage_data):
    f = open(fileName, "w")
    f.write("# vtk DataFile Version 2.0\n")
    f.write("%s \n" % title)
    f.write("ASCII\n")
    f.write("\n")
    f.write("DATASET Damage vector\n")
    f.write("NNODES %d \n" % int(len(damage_data)))   
    f.write("DAMAGE \n")
    for i in range(0, np.shape(damage_data)[0]):
        f.write("{:f} ".format(np.float64(damage_data[i])))
        f.write("\n")
    f.close()