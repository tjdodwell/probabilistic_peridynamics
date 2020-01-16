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
