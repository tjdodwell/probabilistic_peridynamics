////////////////////////////////////////////////////////////////////////////////
//
// opencl_peridynamics.cl
//
// OpenCL Peridynamics kernels
//
// Based on code from Copyright (c) Farshid Mossaiby, 2016, 2017. Adapted for python.
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project

#include "opencl_enable_fp64.cl"

// Macros

#define DPN 3
// MAX_HORIZON_LENGTH, PD_DT, PD_E, PD_S0, PD_NODE_NO, PD_DPN_NODE_NO will be defined on JIT compiler's command line

__kernel void
	InitialValues(
		__global double *Un,
		__global double *Udn
    )
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
		Un[i] = Udn[i] = 0.00;
	}
}

// A horizon by horizon approach is chosen to proceed with the solution, in which
// no assembly of te system of equations is required.

// Update un1
__kernel void
	TimeMarching1(
        __global double *Un1,
		__global double const *Un,
		__global double const *Udn,
		__global int const *BCTypes,
		__global double const *BCValues
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
		Un1[i] = BCTypes[i] == 0 ? Un[i] + PD_DT * (Udn[i]) : Un[i] + BCValues[i];
	}
}


// Calculate force using un1
__kernel void
	TimeMarching2(
        __global double *Udn1,
        __global double const *Un1,
        __global double const *Vols,
		__global int const *Horizons,
		__global double const *Nodes
	)
{
	const int i = get_global_id(0);

	double f0 = 0.00;
	double f1 = 0.00;
	double f2 = 0.00;

	if (i < PD_NODE_NO)
	{
		for (int j = 1; j < MAX_HORIZON_LENGTH; j++)
		{
			const int n = Horizons[MAX_HORIZON_LENGTH * i + j];

			if (n != -1)
			{
				const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later, doesn't need to be done every time
				const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
				const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];


				const double xi_eta_x = Un1[DPN * n + 0] - Un1[DPN * i + 0] + xi_x;
				const double xi_eta_y = Un1[DPN * n + 1] - Un1[DPN * i + 1] + xi_y;
				const double xi_eta_z = Un1[DPN * n + 2] - Un1[DPN * i + 2] + xi_z;

				const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
				const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
                const double y_xi = y - xi;

				const double cx = xi_eta_x / y;
				const double cy = xi_eta_y / y;
				const double cz = xi_eta_z / y;

				const double _E = PD_E;
                const double _A = Vols[i];
				const double _L = xi;

				const double _EAL = -_E * _A / _L;

                f0 += _EAL * cx * y_xi;
                f1 += _EAL * cy * y_xi;
                f2 += _EAL * cz * y_xi;
			}
		}

		Udn1[DPN * i + 0] = f0;
		Udn1[DPN * i + 1] = f1;
		Udn1[DPN * i + 2] = f2;
	}
}

// Update Un
__kernel void
	TimeMarching3(
		__global double *restrict Un,
		__global double *restrict Udn,
		__global double const *restrict Un1,
		__global double const *restrict Udn1
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{

		Un[i] = Un1[i];
		Udn[i] = Udn1[i];
	}
}


__kernel void
	CheckBonds(
		__global int *Horizons,
		__global double const *Un,
		__global double const *Nodes
	)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if ((i < PD_NODE_NO) && (j > 0) && (j < MAX_HORIZON_LENGTH))
	{
		const int n = Horizons[i * MAX_HORIZON_LENGTH + j];

		if (n != -1)
		{
			const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later
			const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
			const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];

			const double xi_eta_x = Un[DPN * n + 0] - Un[DPN * i + 0] + xi_x;
			const double xi_eta_y = Un[DPN * n + 1] - Un[DPN * i + 1] + xi_y;
			const double xi_eta_z = Un[DPN * n + 2] - Un[DPN * i + 2] + xi_z;

			const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
			const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);

			const double s = (y - xi) / xi;

			// Check for state of the bond

			if (s > PD_S0)
			{
				Horizons[i * MAX_HORIZON_LENGTH + j] = -1;  // Break the bond
			}
		}
	}
}

__kernel void
	CalculateDamage(
		__global float *Phi,
		__global int const *Horizons,
		__global int const *HorizonLengths
	)
{
	const int i = get_global_id(0);

	if (i < PD_NODE_NO)
	{
		int active_bonds = 0;

		for (int j = 1; j < MAX_HORIZON_LENGTH; j++)
		{
			if (Horizons[MAX_HORIZON_LENGTH * i + j] != -1)
			{
				active_bonds++;
			}
		}

		Phi[i] = 1.00f - (float) active_bonds / (float) (HorizonLengths[i] - 1);
	}
}