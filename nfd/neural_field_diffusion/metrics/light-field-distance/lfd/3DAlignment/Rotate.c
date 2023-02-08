#include <stdio.h>
#include <math.h>
#include "ds.h"

vector cross(vector v1, vector v2)
{
	vector tmp;
	tmp.x = v1.y*v2.z - v2.y*v1.z;
	tmp.y = v1.z*v2.x - v2.z*v1.x;
	tmp.z = v1.x*v2.y - v2.x*v1.y;

	return tmp;
}

double dot(vector v1, vector v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

vector NewCoor(vector v, double matrix[3][3])
{
	vector	tmp;

	tmp.x =	v.x * matrix[0][0] + v.y * matrix[1][0] + v.z * matrix[2][0];
	tmp.y =	v.x * matrix[0][1] + v.y * matrix[1][1] + v.z * matrix[2][1];
	tmp.z =	v.x * matrix[0][2] + v.y * matrix[1][2] + v.z * matrix[2][2];

	return tmp;
}

vector normalize(vector v)
{
	vector tmp;
	double len;

	len = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);

	tmp.x = v.x / len;
	tmp.y = v.y / len;
	tmp.z = v.z / len;

	return tmp;
}

// rotatation matrix transform e2 to e1
void RotateMatrix(double **matrix, vector e1[2], vector e2[2])
{
	vector	orth[3]={{1,0,0},{0,1,0},{0,0,1}};
	double	d, d1, d2, d3;
	double	matrix1c[3][3], matrixc2[3][3];
	vector	axis1[3], axis2[3];		// x, y, z
	int		i, j, k;
	double	sum;

	// axis of F1
	axis1[0] = normalize( e1[0] );
	axis1[2] = normalize( cross(e1[0], e1[1]) );
	axis1[1] = normalize( cross(e1[0], axis1[2]) );

	// F1 to Fc
	for(i=0; i<3; i++)
	{
		matrix1c[i][0] = axis1[i].x;
		matrix1c[i][1] = axis1[i].y;
		matrix1c[i][2] = axis1[i].z;
	}

	// axis of F2
	axis2[0] = normalize( e2[0] );
	axis2[2] = normalize( cross(e2[0], e2[1]) );
	axis2[1] = normalize( cross(e2[0], axis2[2]) );

	// Fc to F2
	for(i=0; i<3; i++)
	{
		d = dot(axis2[0], cross(axis2[1], axis2[2]));
		d1 = dot(orth[i], cross(axis2[1], axis2[2]));
		d2 = dot(axis2[0], cross(orth[i], axis2[2]));
		d3 = dot(axis2[0], cross(axis2[1], orth[i]));

		matrixc2[i][0] = d1 / d;
		matrixc2[i][1] = d2 / d;
		matrixc2[i][2] = d3 / d;
	}

	for(i=0; i<3; i++)
		for(j=0; j<3; j++)
		{
			sum = 0;

			for(k=0; k<3; k++)
				sum += matrixc2[i][k] * matrix1c[k][j];
					
			matrix[i][j] = sum;
		}

//		tmp2 = NewCoor( NewCoor( NewCoor(data[i], matrixc2), matrix21 ), matrix2c);
// because matrix21 = matrix1c * matrixc2, so
//		tmp2 = NewCoor( NewCoor(data[i], matrixc2), matrix1c);
}

// ***************************************************************************************
// [ x' y' z' 1] = [ x y z 1 ] * M	; [1x4] = [1x4] * [4x4]
void Rotate(pVer vertex, int NumVer, double **matrix)
{
	int			i, j, k;
	double		sum;
	Ver			vTmp;

	for(i=0; i<NumVer; i++)
	{
		for(j=0; j<3; j++)
		{
			sum = matrix[3][j];
			for(k=0; k<3; k++)
				sum += vertex[i].coor[k] * matrix[k][j];

			vTmp.coor[j] = sum;
		}

		vertex[i].coor[0] = vTmp.coor[0];
		vertex[i].coor[1] = vTmp.coor[1];
		vertex[i].coor[2] = vTmp.coor[2];
	}

}

// ****************************************************************************************
// [ x' y' z' 1] = [ x y z 1 ] * M	; [1x4] = [1x4] * [4x4]
void Transform(pVer SrcVer, int NumVer, double matrix[4][4], pVer DestVer)
{
	int			i, j, k;
	double		sum;
	Ver			vTmp;	// vTmp is nessceary, if SrcVer and DestVer are the same

	for(i=0; i<NumVer; i++)
	{
		for(j=0; j<3; j++)
		{
			sum = matrix[3][j];
			for(k=0; k<3; k++)
				sum += SrcVer[i].coor[k] * matrix[k][j];

			vTmp.coor[j] = sum;
		}

		DestVer[i].coor[0] = vTmp.coor[0];
		DestVer[i].coor[1] = vTmp.coor[1];
		DestVer[i].coor[2] = vTmp.coor[2];
	}

}

void RotateX(pVer SrcVer, double T, pVer DestVer, int NumVer)
{
	double		matrix[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

	matrix[0][0] = matrix[1][1] = cos(T);
	matrix[1][0] = -sin(T);
	matrix[0][1] = -matrix[1][0];

	Transform(SrcVer, NumVer, matrix, DestVer);
}

void RotateY(pVer SrcVer, double T, pVer DestVer, int NumVer)
{
	double		matrix[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

	matrix[1][1] = matrix[2][2] = cos(T);
	matrix[2][1] = -sin(T);
	matrix[1][2] = -matrix[2][1];

	Transform(SrcVer, NumVer, matrix, DestVer);
}

void RotateZ(pVer SrcVer, double T, pVer DestVer, int NumVer)
{
	double		matrix[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

	matrix[0][0] = matrix[2][2] = cos(T);
	matrix[2][0] = sin(T);
	matrix[0][2] = -matrix[2][0];

	Transform(SrcVer, NumVer, matrix, DestVer);
}
