from numpy import linalg as LA

def write(fileName, numPoints, x, damage, U):

    f = open(fileName + ".vtu","w")

    f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n")

    f.write("<UnstructuredGrid>\n")
    f.write("\t<Piece NumberOfPoints=\"%s\" NumberOfCells=\"%s\">\n" % (numPoints, numPoints))
    f.write("\t\t<PointData Scalars=\"scalars\">\n")
    f.write("\t\t\t<DataArray type=\"Float32\" Name=\"Damage\" Format=\"ascii\">\n")
    for i in range(0, numPoints):
        f.write("\t\t\t\t %f\n" % damage[i])
    f.write("\t\t\t</DataArray>\n")
    f.write("\t\t\t<DataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
    for i in range(0, numPoints):
        f.write("\t\t\t\t %f %f %f\n" % (U[i,0], U[i,1], U[i,2]))
    f.write("\t\t\t</DataArray>\n")
    f.write("\t\t</PointData>\n")
    f.write("\t\t<Points>\n")
    f.write("\t\t\t<DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
    for i in range(0, numPoints):
        f.write("\t\t\t\t %f %f %f\n" % (x[i,0], x[i,1], x[i,2]))
    f.write("\t\t\t</DataArray>\n")
    f.write("\t\t</Points>\n")
    f.write("\t\t<Cells>\n")
    f.write("\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n")
    for i in range(0, numPoints):
        f.write("\t\t\t\t %s" % i)
    f.write("\t\t\t</DataArray>\n")
    f.write("\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">\n")
    for i in range(0, numPoints):
        f.write("\t\t\t\t %s" % (i + 1))
    f.write("\t\t\t</DataArray>")
    f.write("\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n")
    for i in range(0,numPoints):
        f.write("\t\t\t\t 1\n")
    f.write("\t\t\t</DataArray>\n")
    f.write("\t\t</Cells>\n")
    f.write("\t</Piece>\n")
    f.write("</UnstructuredGrid>\n")
    f.write("</VTKFile>")

    f.close()

def writeParallel(fileName, comm, numPoints, x, damage, U):

    rank = comm.Get_rank();
    nproc = comm.Get_size();

    local_fileName = fileName + "_proc_" + str(rank)

    write(local_fileName,numPoints, x, damage, U)

    if(rank == 0):
        f = open(fileName + ".pvtu","w")
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write("\t<PUnstructuredGrid GhostLevel=\"0\">\n")
        f.write("\t\t<PPoints>\n")
        f.write("\t\t\t<PDataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\"3\"/>\n")
        f.write("\t\t</PPoints>\n")
        f.write("\t\t<PCellData>\n")
        f.write("\t\t</PCellData>\n")
        f.write("\t\t<PPointData Scalars=\"Damage\" Vectors=\"Displacement\">\n")
        f.write("\t\t\t<PDataArray type=\"Float32\" Name=\"Damage\" NumberOfComponents=\"1\"/>\n")
        f.write("\t\t\t<PDataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\"/>\n")
        f.write("\t\t</PPointData>\n")
        for i in range(0,nproc):
            f.write("\t\t<Piece Source=\"" + fileName + "_proc_" + str(i) + ".vtu\"/>\n")
        f.write("\t</PUnstructuredGrid>\n")
        f.write("</VTKFile>\n")
        f.close()
